import time
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "goofybhatoora/banking77-phase-4-evaluation"

api = KaggleApi()
api.authenticate()

print(f"Monitoring Kaggle Kernel: {KERNEL_ID}")
last_status = None

for attempt in range(180):  # Poll every 25 seconds for up to 75 minutes
    try:
        resp = api.kernels_status(KERNEL_ID)
        d = resp.to_dict() if hasattr(resp, "to_dict") else {}
        status = d.get("status") or str(getattr(resp, "status", ""))
        failure_msg = d.get("failureMessage") or getattr(resp, "failure_message", None)

        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status} (failure: {failure_msg})", flush=True)
            last_status = status

        if "COMPLETE" in status.upper():
            print("\n🎉 Kernel completed successfully! Downloading outputs...", flush=True)
            out_dir = ROOT / "eval_output"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                api.kernels_output(KERNEL_ID, path=str(out_dir))
                files = list(out_dir.glob("**/*"))
                print(f"Downloaded {len(files)} files to {out_dir}:", flush=True)
                for f in files:
                    if f.is_file():
                        print(f"  {f.relative_to(out_dir)} ({f.stat().st_size / 1e3:.1f} KB)", flush=True)
            except Exception as e:
                print("Download error:", e, flush=True)
            break
        elif "ERROR" in status.upper():
            print(f"\n❌ Kernel execution failed: {failure_msg}", flush=True)
            out_dir = ROOT / "eval_output"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                api.kernels_output(KERNEL_ID, path=str(out_dir))
                for log_file in out_dir.glob("*.log"):
                    print(f"\n--- {log_file.name} (last 40 lines) ---", flush=True)
                    lines = open(log_file, encoding="utf-8", errors="ignore").readlines()[-40:]
                    print("".join(lines), flush=True)
            except Exception as e:
                print("Could not download error logs:", e, flush=True)
            sys.exit(1)
        else:
            if attempt % 3 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Kernel {KERNEL_ID} is still RUNNING...", flush=True)

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Poll error: {e}", flush=True)

    time.sleep(25)
