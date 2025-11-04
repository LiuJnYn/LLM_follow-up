# -*- coding: utf-8 -*-
import os, json, re
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from utils import Agent
import dashscope
import time
import glob
dashscope.api_key = ""

UNCERTAIN_TOKENS = {"不确定", "不太确定", "有些", "有一点", "说不清", "不好说"}
import matplotlib.pyplot as plt # 绘制图像

from form_data.patient_setting.patient1 import PATIENT_1,PATIENT_2,PATIENT_3


dashscope.api_key = "sk-4c78ba9ff8e94295b93dd8dd3bf99956"
MODEL_NAME = "qwen-plus"
FORM_TYPE = "short_form"  # "short_form" 或 "long_form" 或 "long_form_complex"
PATIENT_TYPE = "3"
PATIENT_INFO = PATIENT_3


def load_form(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_item_id(item: Dict[str, Any], fallback_index: int) -> str:
    return str(item.get("itemID", fallback_index))

def is_parent(item: Dict[str, Any]) -> bool:
    return not (item.get("visible_if") or [])

def question_text(item: Dict[str, Any]) -> str:
    return str(item.get("name", "")).strip()

def extract_labels(item: Dict[str, Any]) -> List[str]:
    return [str(opt["label"]) for opt in (item.get("options") or []) if opt.get("label") is not None]


def build_parent_lists(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    parent_all, type_map = [], {}
    for idx, it in enumerate(items):
        if is_parent(it):
            it = dict(it)
            it["_rt_index"] = idx
            parent_all.append(it)
            t = it.get("type", "other")
            if t not in {"one-choice", "multi-choice", "input-text"}:
                t = "other"
            type_map[len(parent_all) - 1] = t
    return parent_all, type_map


def llm_merge_single_choice(agent: Agent, parent_all: List[Dict[str, Any]], type_map: Dict[int, str]) -> Dict[str, List[int]]:
    parent_questions = [question_text(it) for it in parent_all]
    summary, _= agent.summary_form_info(parent_questions)
    group_text, _= agent.judge_Conbined(summary, parent_questions)

    try:
        raw = json.loads(group_text)
        if not isinstance(raw, dict):
            raise ValueError
    except Exception:
        m = re.search(r"/{[/s/S]*?/}", group_text)
        if not m:
            return {question_text(it): [i] for i, it in enumerate(parent_all) if type_map[i] == "one-choice"}
        raw = json.loads(m.group(0))

    cleaned: Dict[str, List[int]] = {}
    for name, idx_list in raw.items():
        if not isinstance(idx_list, list):
            continue
        kept = [int(i) for i in idx_list if type_map.get(int(i)) == "one-choice"]
        if kept:
            cleaned[name] = kept

    covered = {i for lst in cleaned.values() for i in lst}
    for i, t in type_map.items():
        if t == "one-choice" and i not in covered:
            cleaned[question_text(parent_all[i])] = [i]
    return cleaned


def map_groups_to_item_ids(parent_all: List[Dict[str, Any]], single_choice_groups_idx: Dict[str, List[int]], type_map: Dict[int, str]) -> List[List[str]]:
    idx_to_itemID = {i: pick_item_id(it, it["_rt_index"]) for i, it in enumerate(parent_all)}
    final_groups: List[List[str]] = []

    for _, idxs in single_choice_groups_idx.items():
        final_groups.append([idx_to_itemID[i] for i in idxs])
    for i, t in type_map.items():
        if t in {"multi-choice", "input-text"}:
            final_groups.append([idx_to_itemID[i]])
    return final_groups


def build_id_tag_maps(items):
    id_to_item = {str(it["itemID"]): it for it in items}
    id_to_tag = {str(it["itemID"]): str(it.get("tag", "")) for it in items}
    tag_to_id = {str(it.get("tag", "")): str(it["itemID"]) for it in items if it.get("tag")}
    return id_to_item, id_to_tag, tag_to_id


def check_trigger_conditions(items: list[dict], answered: dict[str, str],
                             id_to_tag: dict[str, str], tag_to_id: dict[str, str]) -> list[str]:
    triggered = []
    answer_view = dict(answered)
    for iid, ans in answered.items():
        tag = id_to_tag.get(iid)
        if tag:
            answer_view[tag] = ans

    for it in items:
        rules = it.get("visible_if") or []
        if not rules:
            continue
        for cond in rules:
            dep_item = str(cond.get("depended_item"))
            dep_opt = str(cond.get("depended_option"))
            if answer_view.get(dep_item) == dep_opt:
                triggered.append(str(it["itemID"]))
                break
    return triggered


def normalize_extracted(chosen_raw, num_questions: int) -> list[list[str]]:
    import json
    data = chosen_raw
    if isinstance(data, tuple) and len(data) >= 1:
        data = data[0]
    if isinstance(data, str):
        s = data.strip()
        try:
            parsed = json.loads(s)
            data = parsed
        except Exception:
            data = [s]
    if data is None:
        data = []
    if isinstance(data, list) and (len(data) == 0 or not isinstance(data[0], list)):
        data = [[str(x)] if x is not None else [] for x in data]
    if len(data) < num_questions:
        data += [[] for _ in range(num_questions - len(data))]
    elif len(data) > num_questions:
        data = data[:num_questions]
    data = [[str(x) for x in sub] for sub in data]
    return data


def is_uncertain(labels) -> bool:
    if labels is None:
        return False
    if isinstance(labels, (int, float, bool)):
        return False
    if isinstance(labels, str):
        tokens = {labels}
    elif isinstance(labels, list):
        tokens = set(map(str, labels))
    else:
        try:
            tokens = set(map(str, labels))
        except Exception:
            return False
    return any(tok in tokens for tok in UNCERTAIN_TOKENS)


def ask_question(agent: Agent, item: Dict[str, Any]) -> str:
    global total_time,total_turns
    q = item["name"]

    labels = extract_labels(item)
    gen_question, gen_timer, gen_token = agent.gen_Conbined_Question([q])
    print("（子问题问答）医生：",gen_question)
    dialogue, res_timer = agent.simulate_patient_response(PATIENT_INFO,gen_question)
    print("患者：", dialogue)
    print("/n")
    ext_result, ext_timer, ext_token = agent.extact_option(dialogue, [q], [labels])
    chosen = normalize_extracted(agent.extact_option(dialogue, [q], [labels]), 1)
    total_time += gen_timer  + ext_timer

    turn_logs.append({
        "natural_question": gen_question,
        "elapsed": gen_timer  + ext_timer,
        "user_answer": dialogue,
        "extracted_answer": chosen[0] if chosen else [],
    })
    time_logs.append(gen_timer  + ext_timer)
    token_logs.append(gen_token + ext_token)

    if is_uncertain(chosen[0]):
        dia_his = "医生：" + gen_question +"患者："+ dialogue 
        follow_question,timer, follw_token = agent.gen_follow_up_question([q],dia_his)
        print(follow_question)
        dialogue, res_timer = agent.simulate_patient_response(PATIENT_INFO,follow_question)
        print("患者：", dialogue)
        print("/n")
        ext_result, ext_timer, ext_token = agent.extact_option(dialogue, [q], [labels])
        turn_logs.append({
            "natural_question": gen_question,
            "elapsed": gen_timer  + ext_timer,
            "user_answer": dialogue,
            "extracted_answer": chosen[0] if chosen else [],
        })
        time_logs.append(gen_timer  + ext_timer)
        token_logs.append(gen_token + ext_token)
        chosen = normalize_extracted(agent.extact_option(dia_his, [q], [labels]), 1)
        total_time += timer + ext_timer
        total_turns += 1
    return ",".join(chosen[0]) if chosen and chosen[0] else "未回答"


def run_interview(agent: Agent, items: List[Dict[str, Any]], final_groups: List[List[str]]):
    global total_turns, total_time
    id_to_item, id_to_tag, tag_to_id = build_id_tag_maps(items)
    answered: dict[str, str] = {}

    for group_ids in final_groups:
        questions = [id_to_item[iid]["name"] for iid in group_ids]
        labels = [extract_labels(id_to_item[iid]) for iid in group_ids]
        print(questions)
        gen_question, gen_timer,gen_tokens = agent.gen_Conbined_Question(questions)
        print("（父问题问答）医生：",gen_question)
        dialogue,res_timer = agent.simulate_patient_response(PATIENT_INFO,gen_question)
        print("患者：", dialogue)
        dia_his = "医生：" + gen_question +"患者："+ dialogue

        chosen_raw,ext_timer,ext_token = agent.extact_option(dia_his, questions, labels)
        chosen_per_q = normalize_extracted(chosen_raw, len(questions))
        turn_logs.append({
            "natural_question": gen_question,
            "elapsed": gen_timer  + ext_timer,
            "user_answer": dialogue,
            "extracted_answer": chosen_per_q if chosen_per_q else [],
        })
        time_logs.append(gen_timer  + ext_timer)
        token_logs.append(gen_tokens + ext_token)
        total_turns += 1
        total_time += gen_timer + ext_timer

        # 不确定追问
        for qi, ch in enumerate(chosen_per_q):
            if is_uncertain(ch):
                dia_his = "医生：" + gen_question +"患者："+ dialogue
                q = questions[qi]
                follow_question,timer,follw_tokens = agent.gen_follow_up_question(qi, dia_his)
                print("医生追问：",follow_question)
                dialogue2, res_timer = agent.simulate_patient_response(PATIENT_INFO,follow_question)
                print("患者：", dialogue2)
                dia_his2 = "医生：" + follow_question +"患者："+ dialogue2
                print("/n")
                chosen_raw,ext_timer,ext_token = agent.extact_option(dia_his, questions, labels)
                chosen_per_q[qi] = normalize_extracted(agent.extact_option(dialogue2, [q], [labels[qi]]), 1)[0]
                total_turns += 1
                total_time += timer + ext_timer

        # 保存父题回答
        for i, iid in enumerate(group_ids):
            answered[iid] = ",".join(chosen_per_q[i]) if chosen_per_q[i] else "未回答"

        # 循环触发子题
        asked_children = set()
        while True:
            newly = check_trigger_conditions(items, answered, id_to_tag, tag_to_id)
            newly = [cid for cid in newly if cid not in answered and cid not in asked_children]
            if not newly:
                break
            for cid in newly:
                ans = ask_question(agent, id_to_item[cid])
                answered[cid] = ans
                asked_children.add(cid)
                total_turns += 1
        print("—— 当前组问答结束 ——/n")

    print("问卷答题结束，回答汇总：")
    for k, v in answered.items():
        print(f"  itemID={k}: {v}")

def save_turn_logs(turn_log):
    """保存每轮对话日志和最终统计摘要到 JSON 文件。"""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"dialogue_turns_{timestamp}.json"
    output_dir = f"long_form/experiment_results/turn_logs/white/{FORM_TYPE}/{PATIENT_TYPE}"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    save_data = {
        "turn_logs": turn_log,   # 每轮详细记录
        "summary": {                   # 附加统计信息
            "总问答轮数": total_turns,
            "总耗时(秒)": round(total_time, 2),
            "平均响应时间(秒/轮)": round(total_time / total_turns, 2) if total_turns else 0,
            "累计token消耗": sum(token_logs)
        }
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    print(f"[结果] 对话回合日志及统计摘要已保存至: {filepath}")

def save_and_plot_results(turn_time,turn_token):
    """在对话结束后，保存响应时间数据并绘制图表。"""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"response_times_{timestamp}.json"
    
    # 确保保存的目录存在
    output_dir = f"long_form/experiment_results/time_data/white/{FORM_TYPE}/{PATIENT_TYPE}"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(turn_time, f, ensure_ascii=False, indent=4)
    print(f"/n[结果] 响应时间数据已保存至: {filepath}")


    if turn_token: 
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(turn_token) + 1), turn_token, marker='s', linestyle='--')
        plt.title('Token Consumption per Dialogue Turn', fontsize=16)
        plt.xlabel('Turn', fontsize=12)
        plt.ylabel('Tokens Consumed', fontsize=12)
        plt.xticks(range(1, len(turn_token) + 1))
        plt.grid(True)

        timestamp_img = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename_img = f"token_consumption_plot_{timestamp_img}.png" # 可以改为 .jpg, .pdf 等
        output_dir_img = f"long_form/experiment_results/plots/white/{FORM_TYPE}/{PATIENT_TYPE}" # 指定图片保存目录
        os.makedirs(output_dir_img, exist_ok=True) # 确保目录存在
        filepath_img = os.path.join(output_dir_img, filename_img)
        plt.savefig(filepath_img, dpi=300) # dpi设置图片分辨率
        print(f"[结果] Token消耗图表已保存至: {filepath_img}")


if __name__ == "__main__":

    for i in range(10):
            # === 初始化全局统计变量（一次性定义） ===
        global turn_logs, time_logs, token_logs, total_time, total_turns
        turn_logs = []
        time_logs = [] 
        token_logs = []
        total_time = 0.0
        total_turns = 0
    # === 获取所有表单文件 ===
        form_dir = r"long_form/form_data/form_json/short_form"
        form_files = glob.glob(os.path.join(form_dir, "*.json"))

        print(f"===检测到 {len(form_files)} 个表单文件：===")
        for path in form_files:
            print("  -", os.path.basename(path))

        # === 循环执行每一个表单 ===
        for idx, file_path in enumerate(form_files, start=1):
            print(f"/n========== 第 {idx} 个表单运行开始：{os.path.basename(file_path)} ==========/n")
            form = load_form(file_path)
            items = form.get("items", [])
            parent_all, type_map = build_parent_lists(items)

            agent = Agent(model=dashscope.Generation.Models.qwen_plus)

            single_choice_groups_idx = llm_merge_single_choice(agent, parent_all, type_map)
            final_groups = map_groups_to_item_ids(parent_all, single_choice_groups_idx, type_map)

            print("最终 itemID 分组：", final_groups)
            run_interview(agent, items, final_groups)
            
        save_and_plot_results(time_logs, token_logs)
        save_turn_logs(turn_logs)
