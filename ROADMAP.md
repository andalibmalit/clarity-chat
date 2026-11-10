# Roadmap

What the next releases hold. Each release has a written plan, is used for a day before it's tagged, and the next is planned only after that, so later entries can change. The current release's changes are in [`CHANGELOG.md`](CHANGELOG.md).

- **v0.2.0, the core.** What a graphical interface will need, in the terminal first: cancel a reply, resume a chat, switch branches, remove marks and replace notes, change a setting for one chat, a chat list, a context size per profile, and `clarity chat --json` so other programs (including other LLMs) can drive every command. Also a key for llama-server, so web pages can't reach it.
- **v0.3.0, the web interface on macOS.** A local web page for the same chats, with themes, sliders for settings, and marks and notes. Every feature still works from the terminal.
- **v0.3.1 to v0.3.3, documents.** Attach a document to a chat, then search it by words, then by meaning, with what was retrieved and where it went in the prompt logged.
- **v0.4.0, setup and models.** A setup step and a short list of models, context presets, replay from the pins in a log, and export and import of chats.
- **v0.5.0, thinking models.** Reasoning stored with each reply, and the extra samplers these models use.
- **v0.6.0, Linux.** CUDA, Vulkan and CPU builds, with the GPU driver and CUDA version logged.
- **v0.7.0, Windows.**
- **1.0.** Cross-platform, and easy to install for people who don't use a terminal.
