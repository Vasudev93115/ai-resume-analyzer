import requests
import pandas as pd
import time

APP_ID = "779ffd02"
APP_KEY = "2370156a066e16a95beb71db418168e7"

jobs = []

headers = {
    "User-Agent": "Mozilla/5.0"
}

for page in range(1, 6):

    url = f"https://api.adzuna.com/v1/api/jobs/in/search/{page}"

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": 50,
        "what": "software developer",
        "content-type": "application/json"
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

        print("Status Code:", response.status_code)

        # Debug API response
        print(response.text[:200])

        if response.status_code != 200:
            print(f"Failed on page {page}")
            continue

        data = response.json()

        for job in data.get("results", []):

            title = job.get("title", "Unknown Role")

            company_info = job.get("company", {})
            company = company_info.get(
                "display_name",
                "Unknown Company"
            )

            description = str(
                job.get("description", "")
            ).strip()

            jobs.append({
                "job_title": f"{title} - {company}",
                "description": description
            })

        time.sleep(1)

    except Exception as e:
        print("Error:", e)

df = pd.DataFrame(jobs)

df.to_csv("jobs.csv", index=False)

print("Jobs collected:", len(jobs))