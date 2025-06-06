# FMAP GUI Automation Script

`fmap_gui_automation.py` automates running FMAP with the Swing GUI to collect
comparable statistics across multiple heuristics. The script uses `pyautogui`
for window control and screenshot capture. Metric extraction and analysis follow
the methodology in `AUTOMATION_AGENT_PROMPT.md`.

## Usage

```bash
pip install -r requirements.txt  # ensure pyautogui and analysis packages
python3 fmap_gui_automation.py
python3 plot_results.py          # generate graphs and summary tables
```

Results will be generated in the `results/` directory along with:

- `experiment_results.csv` – raw experiment data
- `performance_summary.csv` – aggregated statistics
- `experiment_report.md` – summary report

After running the automation script, execute `plot_results.py` to create
visual charts and a `summary_table.csv` that make it easier to review the
collected metrics.

Because the script relies on GUI automation, a graphical environment is required.
