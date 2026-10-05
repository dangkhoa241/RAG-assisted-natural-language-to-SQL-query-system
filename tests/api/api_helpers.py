"""Helpers shared by the API tests."""


def upload(client, content: str, name: str = "data.csv"):
    return client.post("/api/datasets", files={"file": (name, content.encode("utf-8"), "text/csv")})
