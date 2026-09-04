import os
import json
import argparse
import urllib.request
import re

def fetch_github_issues(repo: str, limit: int = 50, state: str = "all"):
    """
    Fetch public issues from GitHub REST API for a given repository (e.g. 'pallets/flask').
    Paginates through multiple pages if limit > 100.
    """
    print(f"[1/3] Fetching up to {limit} issues from GitHub repository: {repo}...")
    issues = []
    page = 1

    while len(issues) < limit:
        per_page = min(limit - len(issues) + 20, 100)
        url = f"https://api.github.com/repos/{repo}/issues?state={state}&per_page={per_page}&page={page}"
        
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "EnergyLandscape-DatasetPipeline/1.0"}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"Error fetching issues from GitHub API (page {page}): {e}")
            break

        if not data:
            break

        for item in data:
            if "pull_request" in item:
                continue

            title = item.get("title", "")
            body = item.get("body", "") or ""
            labels = [lbl["name"] for lbl in item.get("labels", [])]
            file_refs = re.findall(r'`?([a-zA-Z0-9_\-\/]+\.py)`?', body + " " + title)
            
            issues.append({
                "id": f"GH-{item['number']}",
                "number": item["number"],
                "title": title,
                "body": body[:1000],
                "labels": labels,
                "file_references": list(set(file_refs)),
                "html_url": item.get("html_url", "")
            })

            if len(issues) >= limit:
                break

        page += 1

    print(f" -> Fetched {len(issues)} valid issues successfully.")
    return issues

def main():
    parser = argparse.ArgumentParser(description="Fetch GitHub Issues for Energy Landscape Pipeline")
    parser.add_argument("--repo", type=str, default="pallets/flask", help="GitHub repo in format 'owner/repo'")
    parser.add_argument("--limit", type=int, default=20, help="Number of issues to fetch")
    parser.add_argument("--output", type=str, default="dataset_pipeline/raw_issues.json", help="Output file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    issues = fetch_github_issues(args.repo, args.limit)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2)

    print(f"Saved raw issues to {args.output}")

if __name__ == "__main__":
    main()
