# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 이소연
- 리뷰어 : 천세문

# PRT(Peer Review Template)

- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 노트북 전체 셀이 실제로 실행된 상태(`execution_count` 20~31)로 제출되었고, 루브릭 3개 항목의 근거 출력이 모두 남아 있습니다.
    - 근거 1) 최종 비교표 (셀 35 출력) — 기존 KoGPT2 / 원본 SFT / 정제 SFT 3자 비교

      | model | BLEU | ROUGE-L | repetition_rate | distinct-2 |
      |---|---|---|---|---|
      | kogpt2_baseline | 1.088 | 0.0312 | 0.12 | 0.0963 |
      | sft_original_data | 20.232 | 0.1191 | 0.22 | 0.245 |
      | sft_clean_data | 18.713 | 0.17 | 0.16 | 0.255 |

    - 근거 2) SFT 단일 출력 vs RM best-of-4 재정렬 (셀 34 출력)
      ```
      SFT single:         {'avg_rm_score': 1.7962, 'bleu': 18.713, 'rouge_l': 0.17}
      RM best-of-N:       {'avg_rm_score': 3.586,  'bleu': 12.932, 'rouge_l': 0.1252, 'n_candidates': 4}
      RM held-out pairwise accuracy: 0.7496 (random baseline=0.5)
      ```
    - 근거 3) 정제 전/후 EDA 비교 (셀 24 출력) — `RM.best_ranked_low_quality_rate` 0.0926 → 0.0
    - 특히 **RM은 생성기가 아니므로 "RM 적용 결과"를 best-of-N reranking으로 정의**하고 PPO 결과로 표현하지 않겠다고 셀 1에서 명시한 점이 정확합니다. 많이 헷갈리는 부분인데 개념을 정확히 잡고 시작했습니다.

- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 가장 핵심적이라고 본 블럭: `src/reward_model.py`의 `GPTRewardModel` + Bradley-Terry pairwise loss (셀 12). 원본 노트북이 ColossalAI `RewardModelTrainer`에 의존하는 부분을 순수 torch/transformers로 재구현한 지점이라 프로젝트 전체에서 가장 위험 부담이 큰 코드입니다. docstring에 **왜 재구현했는지(torch 1.13.1+cu116 / colossalai 0.2.7 핀이 현재 Colab CUDA 12.x·python 3.11에서 설치되지 않음)** 와 **무엇을 동일하게 유지했는지(GPT2 backbone + scalar reward head, InstructGPT/ColossalAI와 같은 pairwise ranking loss)** 가 같이 적혀 있어서, 코드를 읽지 않고 docstring만으로 의도가 파악됐습니다.
    - 두 번째로 잘 쓰인 주석: `src/prompt_template.py` (셀 6). `AutoTokenizer`가 transformers>=5에서 slow byte-level `GPT2Tokenizer`로 해석되어 한국어가 U+FFFD로 깨진다는 것을 **직접 round-trip 검증한 결과와 함께** 적어두고 `PreTrainedTokenizerFast`로 교체했습니다. "왜 바꿨는가"가 재현 가능한 형태로 남아 있어 리뷰어 입장에서 검증이 쉬웠습니다.
    - `src/train_sft.py`의 `IGNORE_INDEX = -100` 아래 한국어 주석(prompt·padding 토큰을 loss에서 제외하는 이유)도 초심자 관점에서 친절합니다.
    - `src/evaluate.py` docstring 중 "RM이 고른 응답을 같은 RM으로 채점하므로 낙관적(optimistic by construction)" 이라는 자기 한계 명시는 특히 좋았습니다.

- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나 새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 디버깅 기록: 셀 37의 "4. 디버깅 및 문제 해결" 섹션에 ① `src/eda.py` 실행 경로 오류로 `eda_report.json/md`가 생성되지 않아 복사 단계에서 연쇄 `FileNotFoundError` 발생 → 작업 경로 확인 후 경로 수정으로 해결, ② JSON/JSONL 형식 혼동으로 `JSONDecodeError` 발생 → 로더 수정으로 해결 과정이 기록되어 있습니다. 셀 3에서 아예 "실행 전 환경 및 경로 검증" 셀을 만들어 재발을 막은 것도 좋은 대응입니다.
    - 추가 실험 1) **디코딩 설정 7종 비교** (greedy / beam4 / beam8 / top-k50 / top-p0.92 / beam+sampling / beam4_no_rep_penalty)를 `ROUGE-L - 2*repetition_rate` 라는 명시적 기준으로 자동 선택 → `beam8` 채택 (셀 30 출력).
    - 추가 실험 2) **RM 데이터 누수 차단**: ranking triplet을 pair로 펼치기 전에 prompt 단위로 90/10 분할 (`train prompts=9124, val prompts=1013`). 같은 prompt에서 나온 pair가 train/val에 동시에 들어가는 흔한 실수를 미리 막았습니다.
    - 추가 실험 3) **공통 holdout 100개**를 정제 이전에 분리해 base/원본SFT/정제SFT를 완전히 동일한 평가셋에서 비교 (셀 20~21). `[split] excluded 100 shared holdout prompts from SFT and RM before cleaning` 로그로 확인됩니다.
    - 추가 실험 4) 정제로 줄어든 만큼 EDA(Easy Data Augmentation) 스타일로 prompt만 변형해 원본 크기 복원 (`11899 → 10956 → 11899`).

- [x]  **4. 회고를 잘 작성했나요?**
    - 셀 37 "최종 결과 해석 및 제출 결론"에 배운 점·한계가 구체적으로 정리되어 있습니다. 특히 **모든 지표가 같이 오르지 않았다는 사실을 숨기지 않고 해석한 점**이 가장 좋았습니다. 정제 SFT는 BLEU 20.232 → 18.713으로 내려갔지만 ROUGE-L 0.1191 → 0.1700, 반복률 0.22 → 0.16, distinct-2 0.245 → 0.255로 개선되었다는 것을 "표면 일치도와 문맥 겹침·다양성은 다른 것을 측정한다"로 설명했습니다.
    - RM best-of-N에서 reward score는 1.7962 → 3.5860으로 올랐는데 BLEU/ROUGE-L은 오히려 내려간 결과도 "reward 기준과 reference 일치도는 서로 다른 평가 목표"라고 정직하게 해석했습니다. 성능이 올랐다고 과장하지 않은 점이 신뢰가 갑니다.
    - 셀 25에서 augmentation 때문에 completion duplicate_rate가 0.0022 → 0.0798로 상승한 것을 **"오류가 아니라 설계의 결과이지만 특정 답변이 과도하게 반복 학습될 위험은 남는다"** 고 스스로 짚고, 후속 개선 방향(completion 패러프레이징, sample weight 조절)까지 제시한 부분이 회고의 백미였습니다.
    - 셀 13에서 RM train acc 0.9053 vs held-out 0.7496의 약 0.16 격차를 과적합 신호로 읽고 early stopping / weight decay / dropout을 개선안으로 적은 것도 좋습니다.
    - 아쉬운 점: 루브릭에서 요구하는 **전체 코드 실행 플로우 그래프(다이어그램)** 가 없습니다. 아래 회고란에 제안 코드를 남겼습니다.

- [x]  **5. 코드가 간결하고 효율적인가요?**
    - `%%writefile`로 `src/prompt_template.py`, `eda.py`, `clean.py`, `train_sft.py`, `metrics.py`, `decoding_search.py`, `reward_model.py`, `evaluate.py` 8개 모듈로 분리해 노트북 셀에는 실행 커맨드만 남겼습니다. 노트북이 38셀로 유지되어 읽기 매우 편했습니다.
    - 중복 최소화가 잘 되어 있습니다. `clean.py`가 `from eda import korean_ratio, has_mojibake, is_complete_sentence, load_jsonl`로 EDA 판정 함수를 그대로 재사용하고, `metrics.py`의 `bleu / rouge_l / repetition_rate / distinct_n`을 `decoding_search.py`와 `evaluate.py`가 공유합니다. **품질 판정 기준이 EDA와 정제에서 한 곳에만 정의되어 있어 기준이 어긋날 수 없는 구조**입니다.
    - CLI `argparse`로 `--data_path / --output_dir / --eval_path`를 받게 해서 원본·정제 데이터에 같은 스크립트를 재사용합니다 (셀 27, 28이 동일 스크립트, 인자만 다름).
    - `random.seed(230319)`을 `clean.py`, `reward_model.py`에 고정해 재현성을 확보했습니다.
    - PEP8은 대체로 준수합니다(4-space, snake_case, 모듈 docstring, type hint 일부 사용).
    - 개선 제안:
      1. 주석·docstring 언어가 영어와 한국어로 혼재합니다. 제출용이면 한 쪽으로 통일하는 편이 읽기 좋습니다.
      2. 셀 20의 `sys.path.insert(0, 'src')`는 `%%writefile` 구조 때문에 불가피하지만, `src/__init__.py`를 만들거나 `%env PYTHONPATH=src`로 처리하면 더 깔끔합니다.
      3. 셀 24의 before/after 비교 코드가 필드·지표를 하드코딩하고 있어 함수화 여지가 있습니다 (아래 개선 코드 참고).

# 회고(참고 링크 및 코드 개선)

```
[리뷰어 회고]
RM이 답변을 생성하지 않는다는 점을 초반에 명시하고 "RM 적용 결과 = best-of-N reranking"으로
정의한 것이 이 프로젝트의 가장 큰 강점이라고 생각합니다. 저는 RM 결과를 어떻게 보여줘야 할지
정리가 안 됐는데, 이 노트북의 evaluate.py 설계(같은 holdout · 같은 디코딩 설정 고정 후 후보 N개
생성 → RM 채점 → 최고점 선택)를 보고 방향을 잡았습니다.

또 하나 배운 것은 "정제 전에 holdout을 먼저 떼는" 순서입니다. 저는 정제 후에 평가셋을 나눠서
baseline과 clean이 서로 다른 프롬프트로 평가되고 있었는데, 셀 20~21의 순서를 보고 비교가
왜곡되고 있었다는 걸 알았습니다. RM에서 prompt 단위로 먼저 나눈 뒤 pair를 펼치는 것도
같은 맥락에서 꼭 따라해야 할 부분입니다.

지표가 엇갈렸을 때(BLEU↓ / ROUGE-L↑ / 반복률↓) 유리한 것만 고르지 않고 상충 관계를 설명한
회고 태도가 특히 인상적이었습니다.

[참고 링크]
- InstructGPT (Ouyang et al., 2022) — Bradley-Terry pairwise ranking loss 원 논문
  https://arxiv.org/abs/2203.02155
- Hugging Face Text generation strategies — greedy/beam/top-k/top-p 파라미터 정리
  https://huggingface.co/docs/transformers/generation_strategies
- EDA: Easy Data Augmentation (Wei & Zou, 2019) — 셀 21의 증강 기법 근거
  https://arxiv.org/abs/1901.11196

[개선 제안 코드 1] 셀 24의 before/after 비교를 함수화 + 모든 RM completion 필드까지 확장
현재는 'SFT.completion'과 'RM.completion_0'만 하드코딩되어 있어 RM.completion_1 / _2의
변화를 놓칩니다. 아래처럼 리포트에 존재하는 필드를 자동으로 순회하고 델타까지 같이 찍으면
정제 효과를 한눈에 확인할 수 있습니다.

    import json

    METRICS = ['low_korean_ratio_rate', 'sentence_complete_rate',
               'mojibake_rate', 'duplicate_rate']

    def load_report(path):
        """EDA 리포트 json을 {field: stats} 딕셔너리로 읽는다."""
        with open(path, encoding='utf-8') as f:
            return {s['field']: s for s in json.load(f) if 'field' in s}

    def diff_reports(before_path, after_path, metrics=METRICS):
        """정제 전/후 리포트를 비교해 (field, metric, before, after, delta)를 출력한다."""
        before, after = load_report(before_path), load_report(after_path)
        fields = [f for f in before if f in after and 'completion' in f]
        print(f"{'field.metric':<48}{'before':>10}{'after':>10}{'delta':>10}")
        for field in sorted(fields):
            for metric in metrics:
                b, a = before[field].get(metric), after[field].get(metric)
                if b is None or a is None:
                    continue
                print(f"{field + '.' + metric:<48}{b:>10.4f}{a:>10.4f}{a - b:>+10.4f}")

    diff_reports('outputs/eda_report_raw.json', 'outputs/eda_report_clean.json')

이렇게 돌려보면 RM.completion_0의 low_korean_ratio_rate가 0.1366 → 0.1223으로 거의
줄지 않은 것이 드러납니다. 정제 규칙이 "1등 랭크가 저품질이면 triplet 폐기"라서 2·3등에
남아 있는 저품질 응답은 그대로 남기 때문인데, 이건 의도된 설계이므로 회고에 한 줄
"RM은 최고 순위 기준으로만 정제했고 하위 순위 노이즈는 남아 있다"고 적어두면 더 완결됩니다.

[개선 제안 코드 2] 루브릭 4번의 '실행 플로우 그래프' 보완
회고 셀 위에 아래 mermaid 셀 하나만 추가하면 전체 파이프라인이 한눈에 들어옵니다.

    from IPython.display import Markdown, display
    display(Markdown('''
    ```mermaid
    flowchart TD
        A[원본 data_kochatgpt<br/>SFT / RM / PPO] --> B[eda.py<br/>정제 전 품질 측정]
        B --> C[공통 holdout 100개 분리<br/>eval_common.json]
        C --> D[clean.py<br/>필터링 + augmentation]
        D --> E1[train_sft.py<br/>sft_baseline 원본]
        D --> E2[train_sft.py<br/>sft_clean 정제]
        D --> F[reward_model.py<br/>prompt 단위 90/10 분할]
        E2 --> G[decoding_search.py<br/>7개 설정 비교 → beam8]
        E1 --> H[evaluate.py]
        E2 --> H
        F --> H
        G --> H
        H --> I[final_comparison.md<br/>정량표 + 정성 예시]
    ```
    '''))
```
