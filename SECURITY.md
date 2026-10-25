# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| latest release (v1.6.x+) | yes |
| older | no — update via Releases |

## Reporting a vulnerability

EchoLens is a fully offline assistive app: it has no accounts, no network
communication at runtime, and stores nothing beyond its own app directory.

If you still find a security issue (e.g. in model loading, asset parsing or
the foreground service), report it privately:

1. Open a GitHub **security advisory** («Report a vulnerability» on the
   repository Security tab), or
2. Contact the maintainer through the FreedoomForm profile.

Please do not open a public Issue for security problems.

## Scope notes

- `signing/ci-release.jks` is a **public test key** committed on purpose so
  CI builds produce installable APKs. Do not treat it as a secret; replace it
  with your own key for production.
- Model files (`.onnx`, `.tflite`) are loaded from APK assets and the app's
  external files dir; treat third-party model files as untrusted input the
  same way you would treat downloaded media.
