import logging
import json
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple, Literal
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from anthropic import Anthropic
import random
import asyncio
from dataclasses import dataclass, asdict
import os
from dotenv import load_dotenv
import traceback
from enum import Enum
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from aiohttp import web

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Language(str, Enum):
    JAPANESE = "japanese"
    ENGLISH = "english"
    ITALIAN = "italian"

@dataclass
class Config:
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    PORT: int = int(os.getenv("PORT", "8080"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    
    # Construct webhook URL from Render environment variables
    RENDER_EXTERNAL_URL: str = os.getenv("RENDER_EXTERNAL_URL", "")
    WEBHOOK_PATH: str = "/webhook"
    WEBHOOK_URL: str = f"{RENDER_EXTERNAL_URL}{WEBHOOK_PATH}" if RENDER_EXTERNAL_URL else ""

    MAX_WORDS_PER_USER: int = 100
    TARGET_SENTENCES: int = 3
    WORDS_PER_SENTENCE: int = 2
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0

    @classmethod
    def validate_config(cls) -> None:
        required_vars = ["TELEGRAM_TOKEN", "CLAUDE_API_KEY", "MONGODB_URI"]
        
        # Only check RENDER_EXTERNAL_URL in production
        if cls.ENVIRONMENT == "production":
            required_vars.append("RENDER_EXTERNAL_URL")
        
        missing_vars = [
            var for var in required_vars
            if not getattr(cls, var)
        ]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                "Please set them in your .env file"
            )

@dataclass
class UserStats:
    total_texts: int = 0
    remembered_words: Dict[Language, int] = None
    forgotten_words: Dict[Language, int] = None
    translations_requested: int = 0
    explanations_requested: int = 0

    def __post_init__(self):
        if self.remembered_words is None:
            self.remembered_words = {lang: 0 for lang in Language}
        if self.forgotten_words is None:
            self.forgotten_words = {lang: 0 for lang in Language}

@dataclass
class UserState:
    user_id: int
    active_words: Dict[Language, Set[str]]
    stats: UserStats
    current_language: Language = Language.JAPANESE
    last_text: Optional[str] = None
    last_update: datetime = None

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.active_words = {lang: set() for lang in Language}
        self.stats = UserStats()
        self.current_language = Language.JAPANESE
        self.last_text = None
        self.last_update = datetime.now()

class MongoDBService:
    def __init__(self, connection_string: str, database_name: str = "language_learning"):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
        self.users = self.db.users
        self.activity_log = self.db.activity_log

    async def init_indexes(self):
        """Initialize database indexes"""
        await self.users.create_index([("user_id", ASCENDING)], unique=True)
        await self.activity_log.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])

    async def save_user_state(self, user_state: UserState) -> None:
        """Save or update user state"""
        user_data = {
            "user_id": user_state.user_id,
            "active_words": {lang.value: list(words) for lang, words in user_state.active_words.items()},
            "stats": asdict(user_state.stats),
            "current_language": user_state.current_language.value,
            "last_text": user_state.last_text,
            "last_update": datetime.now()
        }
        
        await self.users.update_one(
            {"user_id": user_state.user_id},
            {"$set": user_data},
            upsert=True
        )

    async def load_user_state(self, user_id: int) -> Optional[UserState]:
        """Load user state from database"""
        user_data = await self.users.find_one({"user_id": user_id})
        if not user_data:
            return None

        user_state = UserState(user_id)
        
        for lang in Language:
            words = user_data["active_words"].get(lang.value, [])
            user_state.active_words[lang] = set(words)
        
        stats_data = user_data["stats"]
        user_state.stats = UserStats(
            total_texts=stats_data["total_texts"],
            remembered_words={Language(k): v for k, v in stats_data["remembered_words"].items()},
            forgotten_words={Language(k): v for k, v in stats_data["forgotten_words"].items()},
            translations_requested=stats_data["translations_requested"],
            explanations_requested=stats_data["explanations_requested"]
        )
        
        user_state.current_language = Language(user_data["current_language"])
        user_state.last_text = user_data["last_text"]
        user_state.last_update = user_data["last_update"]
        
        return user_state

    async def log_activity(self, user_id: int, activity_type: str, content: str, language: Optional[Language] = None) -> None:
        """Log user activity"""
        await self.activity_log.insert_one({
            "user_id": user_id,
            "type": activity_type,
            "content": content,
            "language": language.value if language else None,
            "timestamp": datetime.now()
        })

    async def get_recent_activities(self, limit: int = 100) -> List[Dict]:
        """Get recent activities"""
        cursor = self.activity_log.find().sort("timestamp", DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)

    async def close(self):
        """Close MongoDB connection"""
        self.client.close()

class TextGenerator:
    def __init__(self, claude_api_key: str):
        self.client = Anthropic(api_key=claude_api_key)
        self._retries = Config.MAX_RETRIES
        self._retry_delay = Config.RETRY_DELAY

    async def generate_text(self, words: List[str], language: Language) -> str:
        """Generate text using Claude API with random word selection"""
        for attempt in range(self._retries):
            try:
                selected_words = []
                available_words = words.copy()
                
                desired_word_count = random.randint(3, 4) * Config.WORDS_PER_SENTENCE
                while len(selected_words) < desired_word_count and available_words:
                    word = random.choice(available_words)
                    selected_words.append(word)
                    available_words.remove(word)

                logger.info(f"Selected words for text: {selected_words}")
                prompt = self._build_prompt(selected_words, language)
                
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    max_tokens=1000,
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                
                text = response.content[0].text.strip()
                if self._validate_text(text):
                    return text
                
            except Exception as e:
                logger.error(f"Text generation error (attempt {attempt + 1}): {str(e)}")
                if attempt < self._retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise

        raise RuntimeError("Failed to generate valid text after all retries")

    async def translate(self, text: str, language: Language) -> str:
        """Translate text with word explanations"""
        if language == Language.JAPANESE:
            prompt = (
                "For this Japanese text:\n\n"
                f"{text}\n\n"
                "Provide:\n"
                "1. English translation of the full text\n"
                "2. List all words with their readings and meanings in this format:\n"
                "単語 (たんご) - word\n"
                "[one word per line, sorted by appearance]"
            )
        else:
            prompt = (
                f"For this {language.value} text:\n\n"
                f"{text}\n\n"
                "Provide:\n"
                "1. English translation of the full text\n"
                "2. List all notable words/phrases with their meanings:\n"
                "word/phrase - meaning\n"
                "[one per line, sorted by appearance]"
            )
        
        response = await asyncio.to_thread(
            self.client.messages.create,
            max_tokens=1000,
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.content[0].text.strip()

    async def explain(self, text: str, language: Language) -> str:
        """Provide grammatical explanation of the text"""
        if language == Language.JAPANESE:
            prompt = (
                "For this Japanese text:\n\n"
                f"{text}\n\n"
                "Provide:\n"
                "1. List of words with readings (kanji words only)\n"
                "2. Breakdown of sentence structure\n"
                "3. Grammar points used\n"
                "4. Explanation of particles\n"
                "5. Any idioms or special expressions\n"
                "\nFormat each section with clear headings"
            )
        else:
            prompt = (
                f"For this {language.value} text:\n\n"
                f"{text}\n\n"
                "Provide:\n"
                "1. Breakdown of sentence structure\n"
                "2. Grammar points used\n"
                "3. Tenses and moods explained\n"
                "4. Any idioms or special expressions\n"
                "\nFormat each section with clear headings"
            )
        
        response = await asyncio.to_thread(
            self.client.messages.create,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.content[0].text.strip()

    def _build_prompt(self, words: List[str], language: Language) -> str:
        if not words:
            return f"Generate 3-4 natural {language.value} sentences."
        
        return (
            f"Generate 3-4 natural {language.value} sentences using "
            f"these words: {', '.join(words)}\n"
            "Rules:\n"
            "- Use the provided words naturally in the sentences\n"
            "- Make the text coherent and connected\n"
            f"- For Japanese: use appropriate mix of kanji and hiragana\n"
            "- Only respond with the text in the target language"
        )

    def _validate_text(self, text: str) -> bool:
        if not text:
            return False
        sentences = [s.strip() for s in text.split('。' if '。' in text else '.') if s.strip()]
        return 2 <= len(sentences) <= 5

class LanguageLearningBot:
    def __init__(self):
        Config.validate_config()
        self.application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        self.text_generator = TextGenerator(Config.CLAUDE_API_KEY)
        self.db = MongoDBService(Config.MONGODB_URI)
        self.user_states: Dict[int, UserState] = {}
        
        # Add web app with explicit port binding
        self.webapp = web.Application()
        self.webapp.router.add_get("/", self.health_check)
        self.webapp.router.add_post(Config.WEBHOOK_PATH, self.handle_webhook)

        # Register handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.show_help))
        self.application.add_handler(CommandHandler("text", self.get_text))
        self.application.add_handler(CommandHandler("translate", self.translate_text))
        self.application.add_handler(CommandHandler("explain", self.explain_text))
        self.application.add_handler(CommandHandler("stats", self.show_stats))
        self.application.add_handler(CommandHandler("words", self.list_words))
        self.application.add_handler(CommandHandler("language", self.change_language))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_text
        ))
        self.application.add_error_handler(self.error_handler)

    async def setup_commands(self):
        """Set up bot commands in the menu"""
        commands = [
            BotCommand("text", "Get new practice text"),
            BotCommand("translate", "Translate last text"),
            BotCommand("explain", "Get grammatical explanation"),
            BotCommand("stats", "Show learning statistics"),
            BotCommand("words", "List saved words"),
            BotCommand("language", "Change language"),
            BotCommand("help", "Show help message")
        ]
        await self.application.bot.set_my_commands(commands)

    async def handle_error(self, update: Update, error: Exception) -> None:
        """Handle errors in updates"""
        error_message = (
            "❌ Sorry, something went wrong. Please try again later.\n"
            "If the problem persists, contact the bot administrator."
        )
        
        if isinstance(error, ValueError):
            error_message = str(error)
        
        if update and update.effective_message:
            await update.effective_message.reply_text(error_message)

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Global error handler for the bot"""
        logger.error(f"Exception while handling an update: {context.error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Get the exception info
        error = context.error
        try:
            raise error
        except Exception as e:
            await self.handle_error(update, e)

        # Log the error to MongoDB
        try:
            error_data = {
                "timestamp": datetime.now(),
                "update_id": update.update_id if update else None,
                "user_id": update.effective_user.id if update and update.effective_user else None,
                "chat_id": update.effective_chat.id if update and update.effective_chat else None,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            }
            
            # Log error to MongoDB
            await self.db.activity_log.insert_one({
                **error_data,
                "type": "error"
            })
            
        except Exception as e:
            logger.error(f"Failed to log error to database: {e}")
            
    async def health_check(self, request):
        """Simple health check endpoint"""
        return web.Response(text="Bot is running!")

    async def handle_webhook(self, request):
        """Handle incoming webhook updates"""
        try:
            update_data = await request.json()
            update = Update.de_json(update_data, self.application.bot)
            await self.application.process_update(update)
            return web.Response(text="OK")
        except Exception as e:
            logger.error(f"Error processing webhook update: {e}")
            return web.Response(status=500, text="Error processing update")

    async def setup_webhook(self, webhook_url: str):
        """Setup webhook for the bot"""
        webhook_info = await self.application.bot.get_webhook_info()
        
        # Only set webhook if it's not already set to our URL
        if webhook_info.url != webhook_url:
            await self.application.bot.set_webhook(webhook_url)
            logger.info(f"Webhook set to {webhook_url}")

    async def run_app(self):
        """Run the web application with explicit port binding"""
        runner = web.AppRunner(self.webapp)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', Config.PORT)
        await site.start()
        logger.info(f"Web app started on port {Config.PORT}")
        
        # Log the webhook URL
        if Config.WEBHOOK_URL:
            logger.info(f"Webhook URL: {Config.WEBHOOK_URL}")

    async def get_user_state(self, user_id: int) -> UserState:
        """Get user state from cache or database"""
        if user_id not in self.user_states:
            state = await self.db.load_user_state(user_id)
            if state is None:
                state = UserState(user_id)
            self.user_states[user_id] = state
        return self.user_states[user_id]

    async def save_state(self, user_id: int, text: str, activity_type: str, language: Language = None) -> None:
        """Save state to MongoDB"""
        user_state = await self.get_user_state(user_id)
        await self.db.save_user_state(user_state)
        await self.db.log_activity(user_id, activity_type, text, language)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        await self.show_help(update, context)

    async def show_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        user_state = await self.get_user_state(update.effective_user.id)
        help_text = (
            f"🌍 Welcome to Language Learning Bot! (Current: {user_state.current_language.value.capitalize()})\n\n"
            "📝 Commands:\n"
            "/text - Get new practice text\n"
            "/translate - Get translation and word meanings\n"
            "/explain - Get grammatical explanation\n"
            "/stats - Show your learning statistics\n"
            "/words - List your saved words\n"
            "/language - Change learning language\n"
            "/help - Show this help message\n\n"
            "📚 Adding/Removing Words:\n"
            "Simply type or paste words separated by commas,\n"
            "then use the Remember/Forget buttons that appear.\n\n"
            f"Maximum words per language: {Config.MAX_WORDS_PER_USER}"
        )
        await update.message.reply_text(help_text)

    async def change_language(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle language change command"""
        keyboard = [
            [InlineKeyboardButton(lang.value.capitalize(), callback_data=f"lang_{lang.value}")]
            for lang in Language
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Choose your learning language:",
            reply_markup=reply_markup
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle callback queries from inline buttons"""
        query = update.callback_query
        await query.answer()

        if query.data.startswith("lang_"):
            language = Language(query.data[5:])
            user_state = await self.get_user_state(query.from_user.id)
            user_state.current_language = language
            await self.save_state(
                query.from_user.id,
                f"Changed language to {language.value}",
                'language_change',
                language
            )
            await query.edit_message_text(f"Language changed to {language.value.capitalize()}")
            
        elif query.data.startswith(("remember_", "forget_")):
            action, words_text = query.data.split("_", 1)
            words = set(w.lower() for w in words_text.split(","))
            user_state = await self.get_user_state(query.from_user.id)
            current_lang = user_state.current_language
            
            if action == "remember":
                if len(user_state.active_words[current_lang]) + len(words) > Config.MAX_WORDS_PER_USER:
                    await query.edit_message_text(
                        f"❌ This would exceed the maximum of {Config.MAX_WORDS_PER_USER} words.\n"
                        "Please remove some words first."
                    )
                    return
                
                new_words = words - {w.lower() for w in user_state.active_words[current_lang]}
                user_state.active_words[current_lang].update(words)
                user_state.stats.remembered_words[current_lang] += len(new_words)
                
                await self.save_state(
                    query.from_user.id,
                    f"remembered:{','.join(words)}",
                    'word_update',
                    current_lang
                )
                
                await query.edit_message_text(
                    f"✅ Added {len(new_words)} new words.\n"
                    f"📚 Total active words ({current_lang.value}): "
                    f"{len(user_state.active_words[current_lang])}"
                )
                
            elif action == "forget":
                active_words_lower = {w.lower() for w in user_state.active_words[current_lang]}
                forgotten = words & active_words_lower
                
                user_state.active_words[current_lang] = {
                    w for w in user_state.active_words[current_lang]
                    if w.lower() not in words
                }
                
                user_state.stats.forgotten_words[current_lang] += len(forgotten)
                
                await self.save_state(
                    query.from_user.id,
                    f"forgot:{','.join(words)}",
                    'word_update',
                    current_lang
                )
                
                await query.edit_message_text(
                    f"✅ Removed {len(forgotten)} words.\n"
                    f"📚 Total active words ({current_lang.value}): "
                    f"{len(user_state.active_words[current_lang])}"
                )

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle plain text input for words"""
        text = update.message.text.strip()
        words = {w.strip().lower() for w in text.split(",") if w.strip()}
        
        if not words:
            return

        keyboard = [
            [
                InlineKeyboardButton("Remember", callback_data=f"remember_{','.join(words)}"),
                InlineKeyboardButton("Forget", callback_data=f"forget_{','.join(words)}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        user_state = await self.get_user_state(update.effective_user.id)
        await update.message.reply_text(
            f"What would you like to do with these words? ({user_state.current_language.value})\n"
            f"{', '.join(words)}",
            reply_markup=reply_markup
        )

    async def get_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate new text using saved words"""
        try:
            user_state = await self.get_user_state(update.effective_user.id)
            current_lang = user_state.current_language
            
            if not user_state.active_words[current_lang]:
                await update.message.reply_text(
                    f"You haven't added any {current_lang.value} words yet!\n"
                    "Type or paste some words separated by commas."
                )
                return
            
            await update.message.reply_text(f"Generating {current_lang.value} text...")
            text = await self.text_generator.generate_text(
                list(user_state.active_words[current_lang]),
                current_lang
            )
            
            user_state.last_text = text
            user_state.stats.total_texts += 1
            
            await self.save_state(
                update.effective_user.id,
                text,
                'text',
                current_lang
            )
            
            await update.message.reply_text(text)
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            await self.handle_error(update, e)

    async def translate_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Translate the last generated text"""
        try:
            user_state = await self.get_user_state(update.effective_user.id)
            
            if not user_state.last_text:
                await update.message.reply_text(
                    "No text to translate. Use /text first to generate text."
                )
                return
            
            await update.message.reply_text("Translating...")
            translation = await self.text_generator.translate(
                user_state.last_text,
                user_state.current_language
            )
            user_state.stats.translations_requested += 1
            
            await self.save_state(
                update.effective_user.id,
                translation,
                'translation',
                user_state.current_language
            )
            
            await update.message.reply_text(translation)
            
        except Exception as e:
            logger.error(f"Error translating: {str(e)}")
            await self.handle_error(update, e)

    async def explain_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Provide grammatical explanation of the last text"""
        try:
            user_state = await self.get_user_state(update.effective_user.id)
            
            if not user_state.last_text:
                await update.message.reply_text(
                    "No text to explain. Use /text first to generate text."
                )
                return
            
            await update.message.reply_text("Analyzing text...")
            explanation = await self.text_generator.explain(
                user_state.last_text,
                user_state.current_language
            )
            user_state.stats.explanations_requested += 1
            
            await self.save_state(
                update.effective_user.id,
                explanation,
                'explanation',
                user_state.current_language
            )
            
            await update.message.reply_text(explanation)
            
        except Exception as e:
            logger.error(f"Error explaining: {str(e)}")
            await self.handle_error(update, e)

    async def show_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show user statistics"""
        user_state = await self.get_user_state(update.effective_user.id)
        stats = user_state.stats
        
        stats_text = ["📊 Your Learning Statistics:\n"]
        
        for lang in Language:
            active_count = len(user_state.active_words[lang])
            if active_count > 0:
                stats_text.append(f"\n{lang.value.capitalize()}:")
                stats_text.append(f"📝 Active words: {active_count}")
                stats_text.append(f"💭 Words remembered: {stats.remembered_words[lang]}")
                stats_text.append(f"🗑 Words forgotten: {stats.forgotten_words[lang]}")
        
        stats_text.extend([
            f"\n📚 Total texts generated: {stats.total_texts}",
            f"🔄 Translations requested: {stats.translations_requested}",
            f"📖 Explanations requested: {stats.explanations_requested}"
        ])
        
        await update.message.reply_text("\n".join(stats_text))

    async def list_words(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List active words for current language"""
        user_state = await self.get_user_state(update.effective_user.id)
        current_lang = user_state.current_language
        
        if not user_state.active_words[current_lang]:
            await update.message.reply_text(f"No {current_lang.value} words saved yet.")
            return
        
        words_list = sorted(user_state.active_words[current_lang])
        response = (
            f"📚 Your Active {current_lang.value.capitalize()} Words:\n\n"
            f"{', '.join(words_list)}\n\n"
            f"Total: {len(words_list)} words"
        )
        
        await update.message.reply_text(response)
        
    async def init_storage(self) -> None:
        """Initialize MongoDB indexes"""
        try:
            await self.db.init_indexes()
            logger.info("MongoDB indexes initialized")
        except Exception as e:
            logger.error(f"Error initializing MongoDB indexes: {str(e)}")

    async def cleanup(self):
        """Cleanup resources"""
        await self.db.close()

    def run(self) -> None:
        """Run the bot"""
        logger.info("Starting bot...")
        loop = asyncio.get_event_loop()
        
        # Initialize storage and commands
        loop.run_until_complete(self.init_storage())
        loop.run_until_complete(self.setup_commands())
        
        try:
            # For local development
            if Config.ENVIRONMENT == "development":
                self.application.run_polling()
            # For production (Render)
            else:
                if not Config.RENDER_EXTERNAL_URL:
                    raise ValueError("RENDER_EXTERNAL_URL environment variable is required in production")
                
                # Setup web app and webhook
                loop.run_until_complete(self.run_app())
                loop.run_until_complete(self.setup_webhook(Config.WEBHOOK_URL))
                
                # Run forever
                loop.run_forever()
        finally:
            loop.run_until_complete(self.cleanup())

def main():
    try:
        bot = LanguageLearningBot()
        logger.info("Bot started successfully")
        bot.run()
    except Exception as e:
        logger.error(f"Bot startup error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
