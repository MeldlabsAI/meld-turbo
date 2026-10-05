# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting instead: **Security → Report a vulnerability** on this repository. Only the maintainers can see that thread.

Describe what you can reproduce and how. You will get a first reply within a few days.

## Scope

Meld Turbo is a set of patches, scripts and tools around a local llama.cpp server. The server listens on `127.0.0.1` by default; exposing it on a network (`LISTEN=0.0.0.0`) has no authentication, so do that only on a network you trust. Model files are downloaded from Hugging Face and are not part of this repository.
