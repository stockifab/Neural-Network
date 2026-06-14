# Note: this demo was vibe coded with Claude. The neural network itself (src/) was not.

import runpy
import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "demo"))

# Streamlit re-executes this entry script on every rerun, so the app must be
# run fresh each time. `import app` would only run demo/app.py once (it gets
# cached in sys.modules), leaving the page blank on subsequent reruns.
runpy.run_path(str(root / "demo" / "app.py"), run_name="__main__")
