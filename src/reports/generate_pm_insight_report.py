import argparse
from pathlib import Path

import pandas as pd


def clean_text(value, max_len: int = 700) -> str:
    if pd.isna(value):
        return ""

    text = str(value).replace("\n", " ").replace("\r", " ").strip()

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."

    return text


def generate_recommended_action(theme: str, issue: str, keywords: str) -> str:
    theme_lower = str(theme).lower()
    issue_lower = str(issue).lower()
    keywords_lower = str(keywords).lower()
    combined = f"{theme_lower} {issue_lower} {keywords_lower}"

    if "battery" in theme_lower or "charging" in theme_lower:
        return (
            "Investigate battery, charging, charger, and USB-port related failure patterns. "
            "Prioritize QA checks for charging stability, battery health, charger compatibility, and early-life battery failure."
        )

    if "carrier" in theme_lower or "sim" in theme_lower or "network" in theme_lower:
        return (
            "Review product listings and compatibility information for carrier, SIM, LTE/4G, unlocked status, and regional variants. "
            "Improve compatibility warnings and ensure that locked/unlocked device status is accurately described before purchase."
        )

    if "performance" in theme_lower or "lag" in theme_lower or "app stability" in theme_lower:
        return (
            "Analyze performance complaints related to slow response, app crashes, freezing, unexpected shutdowns, and early device failure. "
            "Separate true software lag from hardware failure in future issue labeling to improve triage precision."
        )

    if "quality dissatisfaction" in theme_lower:
        return (
            "Review general quality complaints to identify whether dissatisfaction is driven by build quality, low-end specifications, poor expectations management, or defective devices. "
            "Improve product positioning and listing clarity for lower-cost models."
        )

    if "refurbished" in theme_lower or "damaged condition" in theme_lower or "broken product condition" in theme_lower:
        return (
            "Strengthen inspection standards for refurbished, used, or damaged devices. "
            "Clearly disclose cosmetic defects, missing accessories, battery replacement status, and device condition in product listings."
        )

    if "screen" in theme_lower or "display" in theme_lower:
        return (
            "Inspect screen, display, LCD, and touch-related defect reports. "
            "Check supplier quality, packaging protection, and early-life display failure patterns."
        )

    if "audio" in theme_lower or "speaker" in theme_lower or "call sound" in theme_lower:
        return (
            "Investigate speaker, microphone, call audio, and volume-related complaints. "
            "Consider adding a dedicated audio issue category in the next version of the issue taxonomy."
        )

    if "camera" in theme_lower:
        return (
            "Review camera quality, camera app stability, and photo/video complaints. "
            "Check whether the issue is caused by hardware defects, software compression, camera app instability, or expectation mismatch."
        )

    if "accessory" in theme_lower or "case compatibility" in theme_lower:
        return (
            "Review accessory, case, cover, and compatibility complaints. "
            "Improve product compatibility descriptions and ensure accessories do not block key device functions."
        )

    if "smartwatch" in theme_lower:
        return (
            "Separate smartwatch-related complaints from mobile phone complaints in future preprocessing. "
            "Review smartwatch connectivity, Bluetooth stability, band quality, and feature expectation issues."
        )

    if "service" in theme_lower or "warranty" in theme_lower or "refund" in theme_lower:
        return (
            "Review seller support, return, refund, and warranty workflows. "
            "Identify whether dissatisfaction is driven by product failure, unclear return policies, or after-sales service delays."
        )

    if any(word in combined for word in ["battery", "charge", "charging", "charger", "usb", "port"]):
        return (
            "Investigate battery, charging, charger, and USB-port related failure patterns. "
            "Prioritize QA checks for charging stability, battery health, and accessory compatibility."
        )

    if any(word in combined for word in ["network", "sim", "verizon", "sprint", "unlocked", "locked", "lte", "4g", "carrier"]):
        return (
            "Review product listings and compatibility information for carrier, SIM, LTE/4G, and regional variants. "
            "Improve compatibility warnings and ensure unlocked status is accurately described."
        )

    return (
        "Review representative customer comments to clarify the dominant complaint pattern. "
        "Consider refining the issue taxonomy if this theme appears repeatedly."
    )


def generate_pm_insight_report(
    priority_path: str,
    output_path: str,
    top_n: int = 10,
):
    priority_path = Path(priority_path)
    output_path = Path(output_path)

    if not priority_path.exists():
        raise FileNotFoundError(f"找不到 priority score 文件：{priority_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(priority_path)

    required_cols = [
        "priority_rank",
        "cluster_id",
        "cluster_theme",
        "priority_score",
        "cluster_size",
        "major_issue",
        "top_keywords",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"priority score 文件缺少必要列：{col}")

    df = df.sort_values(by="priority_rank", ascending=True).reset_index(drop=True)

    total_reviews = int(df["cluster_size"].sum())
    total_clusters = len(df)

    top_df = df.head(top_n).copy()

    lines = []

    lines.append("# Product Manager Insight Report")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        "This report summarizes high-priority complaint themes discovered from mobile phone customer reviews. "
        "It combines transformer-based issue prediction, sentence embedding clustering, and priority scoring."
    )
    lines.append("")
    lines.append(f"- Total clustered negative reviews: **{total_reviews}**")
    lines.append(f"- Number of complaint clusters: **{total_clusters}**")
    lines.append(f"- Top priority themes included in this report: **{len(top_df)}**")
    lines.append("")
    lines.append(
        "The priority score is based on cluster size, issue concentration, average model confidence, "
        "and negative review ratio. Since the current public dataset does not provide reliable review timestamps, "
        "short-term growth is represented as a neutral proxy and should be replaced with time-based trend signals in future work."
    )
    lines.append("")

    lines.append("## 2. Top Priority Complaint Themes")
    lines.append("")
    lines.append("| Rank | Theme | Score | Size | Major Issue | Keywords |")
    lines.append("|---:|---|---:|---:|---|---|")

    for _, row in top_df.iterrows():
        rank = int(row["priority_rank"])
        theme = clean_text(row["cluster_theme"], 120)
        score = float(row["priority_score"])
        size = int(row["cluster_size"])
        issue = clean_text(row["major_issue"], 80)
        keywords = clean_text(row["top_keywords"], 180)

        lines.append(
            f"| {rank} | {theme} | {score:.2f} | {size} | {issue} | {keywords} |"
        )

    lines.append("")

    lines.append("## 3. Detailed Priority Analysis")
    lines.append("")

    for _, row in top_df.iterrows():
        rank = int(row["priority_rank"])
        cluster_id = int(row["cluster_id"])
        theme = clean_text(row["cluster_theme"], 200)
        score = float(row["priority_score"])
        size = int(row["cluster_size"])
        issue = clean_text(row["major_issue"], 100)
        keywords = clean_text(row["top_keywords"], 250)

        lines.append(f"### Priority {rank}: {theme}")
        lines.append("")
        lines.append(f"- Cluster ID: **{cluster_id}**")
        lines.append(f"- Priority score: **{score:.2f}**")
        lines.append(f"- Cluster size: **{size}**")
        lines.append(f"- Major issue label: **{issue}**")
        lines.append(f"- Keywords: {keywords}")

        if "issue_concentration" in row:
            try:
                lines.append(f"- Issue concentration: **{float(row['issue_concentration']):.2f}**")
            except Exception:
                pass

        if "avg_issue_confidence" in row:
            try:
                lines.append(f"- Average issue confidence: **{float(row['avg_issue_confidence']):.2f}**")
            except Exception:
                pass

        if "negative_ratio" in row:
            try:
                lines.append(f"- Negative ratio: **{float(row['negative_ratio']):.2f}**")
            except Exception:
                pass

        lines.append("")
        lines.append("Representative customer evidence:")
        lines.append("")

        for i in range(1, 4):
            col = f"representative_review_{i}"
            if col in row and pd.notna(row[col]):
                review = clean_text(row[col], 700)
                if review:
                    lines.append(f"{i}. {review}")

        lines.append("")
        lines.append("Recommended product action:")
        lines.append("")
        lines.append(
            generate_recommended_action(
                theme=theme,
                issue=issue,
                keywords=keywords,
            )
        )
        lines.append("")

    lines.append("## 4. Product Implications")
    lines.append("")
    lines.append(
        "The results suggest that customer dissatisfaction is concentrated around several operationally meaningful themes: "
        "battery and charging reliability, carrier or SIM compatibility, refurbished or damaged device condition, "
        "screen and display defects, performance instability, and audio/call quality issues."
    )
    lines.append("")
    lines.append(
        "For product and operations teams, these results can support prioritization of QA checks, product listing improvements, "
        "seller governance, compatibility warnings, and after-sales support workflows."
    )
    lines.append("")

    lines.append("## 5. Limitations and Future Work")
    lines.append("")
    lines.append(
        "- The current issue classifier is trained as a single-label classifier, so Top-K outputs should be interpreted as candidate rankings rather than true multi-label probabilities."
    )
    lines.append(
        "- Some clusters contain mixed themes because customer reviews often mention several issues in one paragraph."
    )
    lines.append(
        "- The current dataset does not contain reliable review timestamps, so trend growth cannot yet be measured directly."
    )
    lines.append(
        "- Future versions can add multi-label issue detection, time-based trend monitoring, and a dedicated audio/accessory issue category."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== Product Manager insight report 生成完成 ==========")
    print(f"输入文件：{priority_path}")
    print(f"输出文件：{output_path}")
    print(f"总 cluster 数：{total_clusters}")
    print(f"报告展示 top_n：{top_n}")

    print("\nTop priority themes:")
    print(
        top_df[
            [
                "priority_rank",
                "cluster_id",
                "cluster_theme",
                "priority_score",
                "cluster_size",
                "major_issue",
            ]
        ]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--priority_path",
        type=str,
        default="outputs/priority/priority_scores.csv",
        help="priority score 文件路径",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/reports/pm_insight_report.md",
        help="PM insight report 输出路径",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="报告展示的 top priority theme 数量",
    )

    args = parser.parse_args()

    generate_pm_insight_report(
        priority_path=args.priority_path,
        output_path=args.output_path,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()