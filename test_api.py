"""End-to-end test of the REST job cycle: upload -> queue -> poll -> fetch.

Runs the real engine on a tiny config so it finishes quickly.
"""
import io
import time

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)

# the API stores uploads under random names, so we remember them here
C = "api_test_content.jpg"
S = "api_test_style.jpg"


def main():
    # upload the content + style and keep the stored names the API hands back
    names = {}
    for name, src in ((C, "jurica-koletic-portrait.jpg"),
                      (S, "philip-martin-portrait.jpg")):
        with open(src, "rb") as fh:
            r = client.post("/uploads", files={"file": (name, fh, "image/jpeg")})
        print("upload", name, "->", r.status_code, r.json())
        assert r.status_code == 201
        names[name] = r.json()["name"]

    # queue a job referencing the stored names
    req = {
        "content": names[C], "style": names[S],
        "optimizer": "adam", "style_mode": "swd",
        "saliency_guard": "haar",
        "num_steps": 30, "resolution": 256,
    }
    r = client.post("/jobs", json=req)
    print("create job", r.status_code, r.json())
    assert r.status_code == 200
    jid = r.json()["id"]

    # poll until the worker marks it done (or failed)
    for _ in range(120):
        st = client.get(f"/jobs/{jid}").json()
        if st["status"] in ("done", "failed"):
            print("final status:", st["status"])
            if st["status"] == "failed":
                print("ERROR:", st["error"])
                raise SystemExit(1)
            break
        time.sleep(1)
    else:
        raise SystemExit("timed out polling job")

    # the result endpoint should hand back a real image
    r = client.get(f"/jobs/{jid}/result")
    print("result image", r.status_code, "content-type", r.headers.get("content-type"),
          "bytes", len(r.content))
    assert r.status_code == 200 and len(r.content) > 1000
    open("api_test_out.jpg", "wb").write(r.content)
    print("OK: full request/response cycle passed")


main()
