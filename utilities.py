import nvdlib
import json

# (Optional) set your API key to avoid rate limits
# nvdlib.set_api_key("YOUR_NVD_API_KEY")

def fetch_cve_info(cve_id):
    results = nvdlib.searchCVE(cveId=cve_id)
    if not results:
        return None
    r = results[0]
    return {
        "cve_id": r.id,
        "description": next((d.value for d in r.descriptions), ""),
        "cvss_vector": getattr(r, "v31vector", None) or getattr(r, "v30vector", None),
        "cvss_score": getattr(r, "v31score", None) or getattr(r, "v30score", None)
    }


# Example:
if __name__=="__main__":
  info = fetch_cve_info("CVE-2025-8194")
  pretty_info = json.dumps(info, indent=4)
  print(pretty_info)
