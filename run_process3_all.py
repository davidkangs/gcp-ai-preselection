#!/usr/bin/env python3
"""
Process‑3 통합 런처

● infer    : 기본 GUI 추론
● hybrid  : 휴리스틱 + AI 보조 GUI ( --ai-limit N )
● batch   : 배치 추론 다이얼로그 ( --folder PATH … )
● retrain : 수정 세션 재학습 다이얼로그 ( --sessions PATH … )

인자를 주지 않고 실행하면 infer 모드가 자동으로 실행됩니다.
"""

import sys
import argparse
from pathlib import Path
import os

# ──────────────────────────────────────────────
# 공통 경로 세팅 & 폴더 보장
# ──────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
sys.path.append(str(BASE))  # 내부 모듈 import 용

for d in ["sessions", "models", "results", "logs"]:
    (BASE / d).mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 서브 커맨드별 실행 함수
# ──────────────────────────────────────────────

def run_infer(args):
    """기본 GUI 추론"""
    from process3_inference import main
    main()


def run_hybrid(args):
    """휴리스틱 + AI 보조 GUI (AI 추가 개수 제한)"""
    from run_3_hy import apply_total_5_patch  # 기존 패치 함수 재활용

    apply_total_5_patch(limit=args.ai_limit)

    from heuristic_centered_inference import main
    main()


def run_batch(args):
    """배치 추론 다이얼로그"""
    from PyQt5.QtWidgets import QApplication
    from process3_batch import BatchInferenceDialog

    app = QApplication(sys.argv)
    dlg = BatchInferenceDialog(args.folder, args.model, args.mode)
    dlg.exec_()


def run_retrain(args):
    """수정 세션 재학습 다이얼로그"""
    from PyQt5.QtWidgets import QApplication
    from process3_retrain import RetrainingDialog

    app = QApplication(sys.argv)
    dlg = RetrainingDialog(args.sessions, args.model, args.epochs)
    dlg.exec_()


# ──────────────────────────────────────────────
# 메인 엔트리포인트
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process‑3 unified launcher (infer | hybrid | batch | retrain)"
    )

    sub = parser.add_subparsers(dest="cmd")  # required 플래그 제거 → 필수가 아님

    # infer (기본)
    sub.add_parser("infer", help="기본 GUI 추론")

    # hybrid
    p_h = sub.add_parser("hybrid", help="휴리스틱 + AI 보조 GUI")
    p_h.add_argument("--ai-limit", type=int, default=5, help="AI가 추가할 최대 개수")

    # batch
    p_b = sub.add_parser("batch", help="배치 추론")
    p_b.add_argument("--folder", required=True, help="SHP 파일 폴더")
    p_b.add_argument(
        "--model",
        default=str(BASE / "models/true_dqn_model.pth"),
        help="DQN 모델(.pth) 경로",
    )
    p_b.add_argument(
        "--mode", choices=["auto", "district", "road"], default="auto", help="클리핑 모드"
    )

    # retrain
    p_r = sub.add_parser("retrain", help="수정 세션 재학습")
    p_r.add_argument("--sessions", required=True, help="수정 세션 JSON/폴더 경로")
    p_r.add_argument(
        "--model",
        default=str(BASE / "models/true_dqn_model.pth"),
        help="저장할(또는 이어 학습할) 모델 경로",
    )
    p_r.add_argument("--epochs", type=int, default=10, help="학습 epoch 수")

    # 파싱
    args = parser.parse_args()

    # 인자 없이 실행하면 infer 모드로
    if args.cmd is None:
        args.cmd = "infer"

    # 디스패치
    {
        "infer": run_infer,
        "hybrid": run_hybrid,
        "batch": run_batch,
        "retrain": run_retrain,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
