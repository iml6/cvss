import subprocess
import shlex

def search_files(term, path="./cve-data/"):
    matches = set()

    # Search in file contents
    grep_cmd = f'grep -rlI "{shlex.quote(term)}" {shlex.quote(path)}'
    grep_proc = subprocess.run(grep_cmd, shell=True, capture_output=True, text=True)
    for file_path in grep_proc.stdout.splitlines():
        matches.add(file_path)

    # Search in file names
    find_cmd = f'find {shlex.quote(path)} -type f -iname "*{shlex.quote(term)}*"'
    find_proc = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
    for file_path in find_proc.stdout.splitlines():
        matches.add(file_path)

    return sorted(matches)

def cveSearchByYear(year, path="./cve-data/"):
    matches = set()
    term = "CVE-" + str(year)
    # Search in file names
    find_cmd = f'find {shlex.quote(path)} -type f -iname "*{shlex.quote(term)}*"'
    find_proc = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
    for file_path in find_proc.stdout.splitlines():
        matches.add(file_path)

    return sorted(matches)

def cveWithCVSS(fpath):
    find_cmd= f'grep vectorString -m 1 {fpath} -wc'
    find_proc = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
    print(fpath)
    print(find_proc.stdout)




# Example usage
fileList = cveSearchByYear(2020, "./cve-data")
for f in fileList:
    cveWithCVSS(f)
