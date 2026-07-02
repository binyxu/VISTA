# Publishing Checklist

This local repository is already initialized with:

```bash
origin git@github.com:binyxu/VISTA.git
```

## 1. Create the GitHub Repository

Create an empty public repository on GitHub:

```text
owner: binyxu
name:  VISTA
url:   https://github.com/binyxu/VISTA
```

Do not initialize it with a README, license, or `.gitignore`; those files are
already in this local repository.

## 2. Push

From this directory:

```bash
git push -u origin main
```

## 3. Enable GitHub Pages

In the GitHub repository:

1. Open `Settings`.
2. Open `Pages`.
3. Under `Build and deployment`, choose `Deploy from a branch`.
4. Select branch `main`.
5. Select folder `/docs`.
6. Save.

The project page will be available at:

```text
https://binyxu.github.io/VISTA/
```

GitHub Pages may take a few minutes to deploy after the first push.

