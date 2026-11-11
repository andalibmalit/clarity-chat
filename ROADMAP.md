# Roadmap

What the next releases hold. Each release has a written plan, and the next is planned only after it's tagged, so later entries can change. From v0.3.0, the maintainer uses each release as a user would, following its by-hand list, before it's tagged. Released versions and their changes are in [`CHANGELOG.md`](CHANGELOG.md); v0.2.0 built in the terminal the core a graphical interface needs.

- **v0.2.1, sampling and export.** Regenerate a reply several times, each with its own logged seed. Optionally log each token's probability and its likeliest alternatives. Export a conversation as one documented file: turns, token IDs, marks, notes, model pins and settings. The export records what computed each reply, and a check reports what differs when someone recomputes it elsewhere.
- **v0.2.2, steering.** Chat with a model steered by a pinned, logged vector that was made elsewhere.
- **v0.3.0, the web interface on macOS.** A local web page for the same chats, with themes, sliders for settings, and marks and notes. Every feature still works from the terminal.
- **v0.3.1 to v0.3.3, documents.** Attach a document to a chat, then search it by words, then by meaning, with what was retrieved and where it went in the prompt logged.
- **v0.4.0, setup and models.** A one-command install, a setup step and a short list of models, your own model files, context presets, rebuilding a call's setup from the pins in its log, import of chats, and export in Inspect's log format.
- **v0.5.0, thinking models.** Reasoning stored with each reply, and the extra samplers these models use.
- **v0.6.0, Linux.** CUDA, Vulkan and CPU builds, with the GPU driver and CUDA version logged. Check, on a real CUDA and Vulkan run, how the GPU placement check reads llama.cpp's `CUDA_Host` and `Vulkan_Host` buffers.
- **v0.7.0, Windows.**
- **1.0.** Cross-platform, and easy to install for people who don't use a terminal. Before 1.0 is approved, a deep multi-agent review of the whole project (ultrareview).
- **Later, not scheduled.** Live probe readings, when an experiment needs them.
