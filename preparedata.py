import json
import os

def load_discourse_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                try:
                    topic = json.load(f)
                    title = topic.get("title", "")
                    posts = topic.get("post_stream", {}).get("posts", [])
                    content = "\n".join([post.get("cooked", "") for post in posts])
                    data.append({"source": "discourse", "title": title, "content": content})
                except:
                    continue
    return data

def load_tds_pages(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                content = f.read()
                data.append({"source": "tds", "title": filename, "content": content})
    return data

def save_combined_data(data, output_path="combined_data.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    discourse_data = load_discourse_data("discourse.json")
    tds_data = load_tds_pages("tds-pages_md")
    combined = discourse_data + tds_data
    save_combined_data(combined)
    print(f"Combined {len(combined)} documents and saved to combined_data.json")
    