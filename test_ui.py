"""End-to-end test of the htmx UI flow.

Posts a transfer through the form endpoint, then plays the browser's part by
following the polling fragment until it reports done, and finally fetches the
image. Uses a tiny config so it finishes quickly.
"""
import warnings
warnings.filterwarnings("ignore")

import re
import time

from fastapi.testclient import TestClient
import api

client = TestClient(api.app)


def main():
    with open("jurica-koletic-portrait.jpg", "rb") as c, \
         open("Under-the-Wave-off-Kanagawa-1024x691.jpg", "rb") as s:
        files = {
            "content": ("content.jpg", c, "image/jpeg"),
            "style": ("style.jpg", s, "image/jpeg"),
        }
        data = {
            "optimizer": "adam", "style_mode": "swd", "saliency_guard": "haar",
            "num_steps": "15", "resolution": "256",
        }
        r = client.post("/ui/transfers", data=data, files=files)
    print("POST /ui/transfers", r.status_code)
    assert r.status_code == 200
    frag = r.text
    print("polling fragment:", frag[:220].replace("\n", " "))

    # the first fragment should still be polling
    assert "hx-get=\"/ui/job/" in frag, "fragment should keep polling"
    assert "running" in frag

    # walk the polling fragment (the browser would swap it in each time)
    import re
    poll = frag
    for _ in range(90):
        m = re.search(r'hx-get="(/ui/job/[0-9a-f]+)"', poll)
        assert m, "polling fragment lost its hx-get"
        job_url = m.group(1)
        time.sleep(2)
        poll = client.get(job_url).text
        if "done" in poll or "failed" in poll:
            break
    print("final fragment:", poll[:120].replace("\n", " "))
    if "failed" in poll:
        print("POLL ERROR:"); print(poll); raise SystemExit(1)

    # pull the finished image using the id from the poll URL
    job_id = job_url.split("/")[-1]
    img = client.get(f"/ui/job/{job_id}/image")
    print("image", img.status_code, "bytes", len(img.content))
    assert img.status_code == 200 and len(img.content) > 500
    print("OK: htmx UI flow passed")


main()
