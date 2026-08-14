import os
import time
import subprocess

paper_dir = "/Users/adityakinjawadekar/Documents/100xcode/pcems26/paper"
files_to_watch = [
    os.path.join(paper_dir, "norm.tex"),
    os.path.join(paper_dir, "content.tex"),
    os.path.join(paper_dir, "references.bib"),
]

def get_mtimes():
    return [os.path.getmtime(f) for f in files_to_watch if os.path.exists(f)]

def compile_norm():
    print("[Watcher] Compiling norm.pdf...", flush=True)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "norm.tex"], cwd=paper_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["bibtex", "norm"], cwd=paper_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "norm.tex"], cwd=paper_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "norm.tex"], cwd=paper_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[Watcher] Done! norm.pdf updated.", flush=True)

last_mtimes = get_mtimes()
compile_norm()

while True:
    time.sleep(1.5)
    current_mtimes = get_mtimes()
    if current_mtimes != last_mtimes:
        last_mtimes = current_mtimes
        compile_norm()
