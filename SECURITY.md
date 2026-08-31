# Security Policy

## Supported versions

Security fixes target the latest version on the default branch. This project is an actively changing research prototype, so deployments should pin a reviewed commit and dependencies.

## Reporting a vulnerability

Please do not disclose exploitable details in a public issue. Use GitHub's private vulnerability reporting for this repository if enabled; otherwise contact the maintainer privately through the repository owner's GitHub profile with:

- a concise description and impact;
- affected file, endpoint, or configuration;
- reproducible steps or a minimal proof of concept;
- a suggested mitigation, if known.

Never include API keys, personal data, or private research content in a report.

The default development API is not an authenticated public service. Put it behind authentication and a restrictive reverse proxy before exposing it to untrusted users.

