# ChatGPT Telegram Bot: **GPT-4. Fast. No daily limits. Special chat modes**

<div align="center">
<img src="https://raw.githubusercontent.com/saferoni/2p3bot_test/main/static/header.png" align="center" style="width: 100%" />
</div>

We all love [chat.openai.com](https://chat.openai.com), but... It's TERRIBLY laggy, has daily limits, and is only accessible through an archaic web interface.

This repo is ChatGPT re-created as Telegram Bot.

## 🤑 Payments
[My bot] supports many payments providers:
- 💎 Crypto
- [Stripe](https://stripe.com)
- [Smart Glocal](https://smart-glocal.com)
- [Unlimint](https://www.unlimint.com)
- [ЮMoney](https://yoomoney.ru)
- and [many-many other](https://core.telegram.org/bots/payments#supported-payment-providers)

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

4. Edit `config/config.example.yml` to set your tokens and run 2 commands below (*if you're advanced user, you can also edit* `config/config.example.env`):
    ```bash
    mv config/config.example.yml config/config.yml
    mv config/config.example.env config/config.env
    ```

5. 🔥 And now **run**:
    ```bash
    docker-compose --env-file config/config.env up --build
    ```
