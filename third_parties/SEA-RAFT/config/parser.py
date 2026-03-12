import json
import argparse

def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

# def parse_args(parser):
#     entry = parser.parse_args()
#     json_path = entry.cfg
#     args = json_to_args(json_path)
#     args_dict = args.__dict__
#     for index, (key, value) in enumerate(vars(entry).items()):
#         args_dict[key] = value
#     return args

# ----------------- fix -----------------
def parse_args(entry_ns: argparse.Namespace):
    """
    entry_ns : 已经解析好的 Namespace（含 --cfg / --path）
    返回：合并 JSON + CLI 后的新 Namespace
    """
    # 1) 读取 JSON -> args_json
    json_path = entry_ns.cfg
    args_json = json_to_args(json_path)      # Namespace from JSON

    # 2) CLI 覆盖 JSON，同名键优先用 CLI
    merged = args_json.__dict__
    merged.update(vars(entry_ns))            # 覆写

    return argparse.Namespace(**merged)
# ---------------------------------------
