# ChatGPT Telegram Bot: **GPT-4. Fast. No daily limits. Special chat modes**

<div align="center">
<img src="https://raw.githubusercontent.com/saferoni/2p3bot_test/main/static/header.png" align="center" style="width: 100%" />
</div>

We all love [chat.openai.com](https://chat.openai.com), but... It's TERRIBLY laggy, has daily limits, and is only accessible through an archaic web interface.

This repo is ChatGPT re-created as Telegram Bot. **And it works great.**

You can deploy your own bot, or use mine: [@chatgpt_karfly_bot](https://t.me/chatgpt_karfly_bot)

## Features
- Low latency replies (it usually takes about 3-5 seconds)
- No request limits
- Message streaming (watch demo)
- GPT-4 support
- Group Chat support (/help_group_chat to get instructions)
- DALLE 2 (choose 👩‍🎨 Artist mode to generate images)
- Voice message recognition
- Code highlighting
- 15 special chat modes: 👩🏼‍🎓 Assistant, 👩🏼‍💻 Code Assistant, 👩‍🎨 Artist, 🧠 Psychologist, 🚀 Elon Musk and other. You can easily create your own chat modes by editing `config/chat_modes.yml`
- Support of [ChatGPT API](https://platform.openai.com/docs/guides/chat/introduction)
- List of allowed Telegram users
- Track $ balance spent on OpenAI API

---

## 🤑 Payments
[My bot] supports many payments providers:
- 💎 Crypto
- [Stripe](https://stripe.com)
- [Smart Glocal](https://smart-glocal.com)
- [Unlimint](https://www.unlimint.com)
- [ЮMoney](https://yoomoney.ru)
- and [many-many other](https://core.telegram.org/bots/payments#supported-payment-providers)

If you want to add payments to your bot and create profitable business – write me on Telegram ([@karfly](https://t.me/karfly)).

## Bot commands
- `/retry` – Regenerate last bot answer
- `/new` – Start new dialog
- `/mode` – Select chat mode
- `/balance` – Show balance
- `/settings` – Show settings
- `/help` – Show help

## Setup
1. Get your [OpenAI API](https://openai.com/api/) key

2. Get your Telegram bot token from [@BotFather](https://t.me/BotFather)

3. Edit `config/config.example.yml` to set your tokens and run 2 commands below (*if you're advanced user, you can also edit* `config/config.example.env`):
    ```bash
    mv config/config.example.yml config/config.yml
    mv config/config.example.env config/config.env
    ```

4. 🔥 And now **run**:
    ```bash
    docker-compose --env-file config/config.env up --build
    ```
