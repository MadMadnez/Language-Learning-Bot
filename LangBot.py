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
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import traceback
from enum import Enum

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
    STORAGE_CHAT_ID: str = os.getenv("STORAGE_CHAT_ID", "")
    MAX_WORDS_PER_USER: int = 100
    TARGET_SENTENCES: int = 3  # Target number, but not strict
    WORDS_PER_SENTENCE: int = 2
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0

    @classmethod
    def validate_config(cls) -> None:
        missing_vars = [
            var for var in ["TELEGRAM_TOKEN", "CLAUDE_API_KEY", "STORAGE_CHAT_ID"]
            if not getattr(cls, var)
        ]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                "Please set them in your .env file"
            )

class TextGenerator:
    def __init__(self, claude_api_key: str):
        self.client = Anthropic(api_key=claude_api_key)
        self._retries = Config.MAX_RETRIES
        self._retry_delay = Config.RETRY_DELAY

    async def generate_text(self, words: List[str], language: Language) -> str:
        """Generate text using Claude API with random word selection"""
        for attempt in range(self._retries):
            try:
                # Randomly select words for each sentence
                selected_words = []
                available_words = words.copy()
                
                # Select random words for approximately 3-4 sentences
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
        return 2 <= len(sentences) <= 5  # Allow some flexibility

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

class TelegramStorage:
    def __init__(self, bot: 'LanguageLearningBot', storage_chat_id: str):
        self.bot = bot
        self.storage_chat_id = storage_chat_id
        
    async def save_state(self, user_id: int, text: str, message_type: str, language: Language = None) -> None:
        """Save state to Telegram chat"""
        data = {
            'user_id': user_id,
            'type': message_type,
            'content': text,
            'language': language.value if language else None,
            'timestamp': datetime.now().isoformat()
        }
        await self.bot.application.bot.send_message(
            self.storage_chat_id,
            f"DATA:{json.dumps(data)}"
        )

    async def load_messages(self, limit: int = 100) -> List[Dict]:
        """Load recent messages from storage chat"""
        messages = []
        async for message in self.bot.application.bot.get_chat_history(
            self.storage_chat_id,
            limit=limit
        ):
            if message.text and message.text.startswith("DATA:"):
                try:
                    data = json.loads(message.text[5:])
                    messages.append(data)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse message: {message.text}")
        return messages

class LanguageLearningBot:
    def __init__(self):
        Config.validate_config()
        self.application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        self.text_generator = TextGenerator(Config.CLAUDE_API_KEY)
        self.storage = TelegramStorage(self, Config.STORAGE_CHAT_ID)
        self.user_states: Dict[int, UserState] = {}

        # Register handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.show_help))
        self.application.add_handler(CommandHandler("text", self.get_text))
        self.application.add_handler(CommandHandler("translate", self.translate_text))
        self.application.add_handler(CommandHandler("explain", self.explain_text))
        self.application.add_handler(CommandHandler("stats", self.show_stats))
        self.application.add_handler(CommandHandler("words", self.list_words))
        self.application.add_handler(CommandHandler("language", self.change_language))
        
        # Add handlers for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Add handler for plain text (for word input)
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_text
        ))
        
        self.application.add_error_handler(self.error_handler)

    def get_user_state(self, user_id: int) -> UserState:
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState(user_id)
        return self.user_states[user_id]

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

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.show_help(update, context)

    async def show_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_state = self.get_user_state(update.effective_user.id)
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
                user_state = self.get_user_state(query.from_user.id)
                user_state.current_language = language
                await query.edit_message_text(f"Language changed to {language.value.capitalize()}")
                
            elif query.data.startswith(("remember_", "forget_")):
                action, words_text = query.data.split("_", 1)
                # Convert to lowercase when processing
                words = set(w.lower() for w in words_text.split(","))
                user_state = self.get_user_state(query.from_user.id)
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
                    
                    await self.storage.save_state(
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
                    # Convert both sets to lowercase for comparison
                    active_words_lower = {w.lower() for w in user_state.active_words[current_lang]}
                    forgotten = words & active_words_lower
                    
                    # Remove words case-insensitively
                    user_state.active_words[current_lang] = {
                        w for w in user_state.active_words[current_lang]
                        if w.lower() not in words
                    }
                    
                    user_state.stats.forgotten_words[current_lang] += len(forgotten)
                    
                    await self.storage.save_state(
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
        # Convert to lowercase when creating the set
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
        
        user_state = self.get_user_state(update.effective_user.id)
        await update.message.reply_text(
            f"What would you like to do with these words? ({user_state.current_language.value})\n"
            f"{', '.join(words)}",
            reply_markup=reply_markup
        )

    async def get_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            user_state = self.get_user_state(update.effective_user.id)
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
            
            await self.storage.save_state(
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
        try:
            user_state = self.get_user_state(update.effective_user.id)
            
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
            
            await self.storage.save_state(
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
        try:
            user_state = self.get_user_state(update.effective_user.id)
            
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
            
            await self.storage.save_state(
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
        user_state = self.get_user_state(update.effective_user.id)
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
        user_state = self.get_user_state(update.effective_user.id)
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

    async def handle_error(self, update: Update, error: Exception) -> None:
        error_message = (
            "❌ Sorry, something went wrong. Please try again later.\n"
            "If the problem persists, contact the bot administrator."
        )
        
        if isinstance(error, ValueError):
            error_message = str(error)
        
        if update and update.effective_message:
            await update.effective_message.reply_text(error_message)

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error(f"Exception while handling an update: {context.error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await self.handle_error(update, context.error)

    async def init_storage(self) -> None:
        """Initialize storage from Telegram chat history"""
        try:
            messages = await self.storage.load_messages()
            for msg in messages:
                user_id = msg['user_id']
                if user_id not in self.user_states:
                    self.user_states[user_id] = UserState(user_id)
                
                language = Language(msg.get('language', 'japanese'))
                
                if msg['type'] == 'word_update':
                    content = msg['content']
                    action, words = content.split(':', 1)
                    words = {w.strip() for w in words.split(',')}
                    
                    if action == 'remembered':
                        self.user_states[user_id].active_words[language].update(words)
                        self.user_states[user_id].stats.remembered_words[language] += len(words)
                    elif action == 'forgot':
                        self.user_states[user_id].stats.forgotten_words[language] += len(words)
                
                elif msg['type'] == 'text':
                    self.user_states[user_id].stats.total_texts += 1
                elif msg['type'] == 'translation':
                    self.user_states[user_id].stats.translations_requested += 1
                elif msg['type'] == 'explanation':
                    self.user_states[user_id].stats.explanations_requested += 1
            
            logger.info("Storage initialized from Telegram chat history")
            
        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")

    def run(self) -> None:
        """Run the bot"""
        logger.info("Starting bot...")
        asyncio.get_event_loop().run_until_complete(self.init_storage())
        asyncio.get_event_loop().run_until_complete(self.setup_commands())
        self.application.run_polling()

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
