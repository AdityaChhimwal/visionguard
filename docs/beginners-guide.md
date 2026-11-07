# VisionGuard beginner's guide

## Find and open the project

Your working GitHub clone is at **`D:\good_projects\visionguard`**. In File Explorer, click the address bar, paste that path, and press **Enter**. In VS Code, select **File → Open Folder**, paste the same path, and select **Open**. Choose **Terminal → New Terminal** and select **Git Bash**.

Use this folder for everyday work. The previous locations are Windows directory junctions pointing to this same folder: they preserve compatibility with existing editor sessions without keeping separate working copies. Future projects belong under `D:\good_projects` unless you choose a different location.

## Run the app on your PC

The project has its Python environment and the 15-category model bundle installed. In the VS Code Git Bash terminal run:

```bash
cd /d/good_projects/visionguard
bash start-local.sh
```

Keep the terminal open while using [the local app](http://127.0.0.1:8000/). Initial model checks can take a little time. Choose a category, click an example or select a matching image, and click **Analyze Image**. Press **Ctrl+C** in the terminal to stop the server.

The app does not automatically start when Windows starts. Run the commands again whenever you need it. The [local readiness page](http://127.0.0.1:8000/ready) lists loaded categories once startup finishes. The launcher finds the repository relative to its own file, uses its `.venv`, and requires all 15 models.

Git Bash writes the D: drive as `/d`; `~` points to your Windows home folder on C:. Use `/d/good_projects/visionguard` when changing into this project.

## Move the project or set up another PC

Git history, models, datasets, and earlier verification records have been preserved in the working folder. The local Python environments were rebuilt for its new location. The additional verification records from the former everyday clone are in `outputs/mvtec-rollout-v1`. Complete recovery copies of both original folders are retained under `D:\good_projects\_relocation_backups\visionguard-20261107`, in `codex-workspace` and `local-clone`. Use those copies for recovery and `D:\good_projects\visionguard` for everyday work. Rebuild a Python environment before using a restored backup.

If you move it again, stop the server, rename the moved `.venv` directory to `.venv-before-move`, then run `uv sync --locked --python 3.13` to create a fresh environment at the new location. Python environments can contain absolute paths, so copying one is insufficient. Start with `bash start-local.sh` after the models are in `artifacts`. Avoid using two independent clones for everyday editing.

On another PC, install Git, VS Code, Python 3.13, and uv, then run:

```bash
mkdir -p /d/good_projects
cd /d/good_projects
git clone https://github.com/AdityaChhimwal/visionguard.git
cd visionguard
uv sync --locked --python 3.13
```

Next install the prepared model bundle using the [README commands](../README.md#run-with-a-prepared-artifact). A Git clone supplies the source and history; model archives, datasets, and Python environments are separate local files. Choose a different drive path if that PC has no D: drive.

Moving the local folder does not change GitHub's repository URL or Render's service. Render builds from GitHub using repository-relative paths; it does not read your Windows folder.

## Understand the folders

- `visionguard/web` contains the page, styles, and browser interactions.
- `visionguard/api` contains the FastAPI routes and model registry.
- `visionguard/ml` contains preprocessing, model loading, and heatmap rendering.
- `configs` defines supported categories and model settings.
- `scripts` contains preparation, evaluation, and verification commands.
- `docs` contains the architecture, benchmark results, and interview guide.
- `artifacts` contains locally installed models and examples. These files are excluded from Git and must be provisioned separately on a fresh clone.
- `.venv` contains the local Python environment. Git does not transfer it.

## Make and save changes

Open a file in VS Code, edit it, and press **Ctrl+S**. Restart the local server after Python or configuration changes. Refresh the browser after frontend changes.

Use these commands in Git Bash to inspect your work:

```bash
git status
git diff
```

To save an intentional change in Git and upload it to GitHub:

```bash
git add path/to/changed-file
git commit -m "Describe the change"
git push
```

Replace the example path and message with your own. Run `git status` before adding files and review the list. Avoid adding model files, datasets, credentials, or personal files. The repository's `.gitignore` excludes its normal local model and data folders.

To get changes already merged on GitHub, first check `git status`, then run `git pull --ff-only` when your working tree is clean. A fresh clone needs the separate model installation described in the [README](../README.md).

## Check the project

From the repository folder, run the lightweight checks:

```bash
uv sync --locked --python 3.13
uv run --no-sync ruff check .
uv run --no-sync python -m pytest -m "not ml" -q
```

The GitHub repository's **Actions** tab also shows automated checks. For model preparation and benchmark commands, see the [README](../README.md) and [benchmark report](mvtec-ad-results.md).

## Manage the hosted app on Render

Open the [Render dashboard](https://dashboard.render.com/) and sign in to the account that owns VisionGuard. Open the **visionguard** web service. Its public address is [visionguard-ljwy.onrender.com](https://visionguard-ljwy.onrender.com).

- **Check status and logs.** Open the service's **Events** and **Logs** pages. Build logs help diagnose installation or model download failures. Runtime logs show startup and readiness problems.
- **Check readiness.** Open the [public readiness page](https://visionguard-ljwy.onrender.com/ready). It should list all 15 categories after startup.
- **Deploy an update.** Changes to application files must first reach GitHub. The Blueprint sets `autoDeployTrigger: off`, so open the service and use **Manual Deploy** to deploy the desired commit. Changes to the Blueprint configuration may also require a Blueprint sync, depending on its Auto Sync setting. Verify the commit and service status in Render afterward.
- **Model assets.** The Blueprint pins the release archive URL and SHA-256 in `render.yaml`. A replacement model bundle needs a verified release and matching checksum before a deployment. The [deployment guide](deployment.md) explains the settings.
- **Free service.** Render can spin down after inactivity. The first visit can take longer while it starts and verifies the models. Check the plan and usage in your dashboard before changing any paid settings.
- **Rollback.** Render's deploy history offers rollback where available. Select a previously successful deploy and verify readiness and examples afterward.

After deploying, open the public app in a private browser window. Test a normal and a defective example for an object and a texture. Check their heatmaps and confirm the app reports an error for an invalid image.

## Common problems

- **The browser cannot connect to localhost.** Start the server in the project folder and wait for the startup message. Keep its terminal open.
- **Port 8000 is already in use.** An existing server may already be running. Open [localhost:8000](http://127.0.0.1:8000/) first. If needed, stop the server in the terminal where it was started.
- **Categories are missing or readiness fails.** Check that `artifacts` contains all 15 model directories and `artifacts/samples` contains the examples. Read the terminal's startup error and follow the prepared-bundle instructions in the [README](../README.md).
- **`uv` is not recognized.** Reopen VS Code after installing it. The local startup command above runs the project's Python directly. See the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
- **Git refuses to pull.** Run `git status` and review your changes. Commit or otherwise preserve work you need before updating. Ask for help if a conflict appears.
- **The public app loads slowly.** Wait for Render to wake, then check the service logs and readiness page.

The [interview guide](interview-guide.md) explains the model and demo. The [benchmark report](mvtec-ad-results.md) includes the measured errors as well as the scores.
