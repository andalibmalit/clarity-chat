# Roadmap

What the next releases hold. Each release has a written plan, so later entries can change. From v0.2.3 to 1.0 the releases are built in one sprint: each gets its plan, reviews and QA pass, and the maintainer uses the whole as a user would before the final review and 1.0, rather than each release before its tag. Released versions and their changes are in [`CHANGELOG.md`](CHANGELOG.md); v0.2.0 built in the terminal the core a graphical interface needs, v0.2.1 added resampling, token probabilities and export, v0.2.2 added steering on the Transformers backend, and v0.2.3 the commands the web interface reads: comparing replies, every branch, any reply by its call ID, and where things are mid-reply.

- **v0.3.0, the web interface on macOS.** A local web page for the same chats, with themes, sliders for settings, and marks and notes. Every feature still works from the terminal.
- **v0.3.1 to v0.3.3, documents.** Attach a document to a chat, then search it by words, then by meaning, with what was retrieved and where it went in the prompt logged.
- **v0.4.0, setup and models.** A one-command install, a setup step and a short list of models, your own model files, context presets, rebuilding a call's setup from the pins in its log, import of chats, and export in Inspect's log format.
- **v0.5.0, thinking models.** Reasoning stored with each reply, and the extra samplers these models use.
- **v0.6.0, Linux.** CUDA, Vulkan and CPU builds, with the GPU driver and CUDA version logged. Check, on a real CUDA and Vulkan run, how the GPU placement check reads llama.cpp's `CUDA_Host` and `Vulkan_Host` buffers.
- **v0.7.0, Windows.**
- **1.0.** Cross-platform, and easy to install for people who don't use a terminal. Before 1.0, a full-project review by parallel agents, each finding independently checked, plus a QA pass.
- **Later, not scheduled.** Live probe readings, when an experiment needs them. Steering on llama.cpp through its control vectors, and comparing a steered reply across the two backends.
