"""
share.py – Launch GasWatch and create a public ngrok tunnel.
Your friend can paste the printed URL into any browser, any network.

Usage:
    python share.py
    python share.py --token YOUR_NGROK_TOKEN   (needed once to unlock 8-hour sessions)
"""

import argparse
import subprocess
import sys
import time
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="ngrok authtoken (one-time setup, from https://dashboard.ngrok.com/)")
    parser.add_argument("--port", type=int, default=8506)
    args = parser.parse_args()

    from pyngrok import ngrok, conf

    # Save authtoken if provided
    if args.token:
        conf.get_default().auth_token = args.token
        ngrok.set_auth_token(args.token)
        print(f"ngrok token saved.")

    port = args.port

    # Start Streamlit in background
    print(f">> Starting GasWatch on port {port}...")
    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         os.path.join(os.path.dirname(__file__), "app.py"),
         "--server.port", str(port),
         "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give Streamlit a moment to start
    time.sleep(3)

    # Open ngrok tunnel
    print("Opening public tunnel via ngrok...")
    try:
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        print("\n" + "=" * 60)
        print("  GasWatch is LIVE!")
        print(f"  Share this URL with your friend:")
        print(f"\n     --> {public_url}\n")
        print("  Works on any device, any network.")
        print("  Press Ctrl+C to stop.\n" + "=" * 60)

        # Keep running until Ctrl+C
        streamlit_proc.wait()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[!] ngrok error: {e}")
        print("\nIf you see 'authtoken' errors, run:")
        print("    python share.py --token YOUR_TOKEN")
        print("Get a free token at: https://dashboard.ngrok.com/signup\n")
    finally:
        ngrok.kill()
        streamlit_proc.terminate()
        print("Stopped.")

if __name__ == "__main__":
    main()
