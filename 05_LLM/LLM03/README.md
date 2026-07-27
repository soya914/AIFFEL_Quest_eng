# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 이소연
- 리뷰어 : 임성배


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 최종 결과물이 산출될 수 있는 코드 제출을 확인하였습니다.  
        <img width="852" height="1147" alt="image" src="https://github.com/user-attachments/assets/1d0b89ed-7033-43b2-bdf4-910002cc05cc" />

    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - Advanced RAG를 통해 얻은 평가 지표 상승이 우연이었는지 유의미한지 검증을 위해 문항을 늘려서 검증한 단계이며, 주석과 마크다운을 통해 통계검정의 흐름을 짚어주어서 확인하기 용이하였습니다.  
        <img width="823" height="911" alt="image" src="https://github.com/user-attachments/assets/0ba0b1a0-c31b-4d9a-b513-92cb4c63d43f" />

        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - Batch 적재를 통해 오류 사전 방지  
        <img width="955" height="836" alt="image" src="https://github.com/user-attachments/assets/78b47e50-0e03-478a-b5f6-994172bdfe5c" />

        
- [X]  **4. 회고를 잘 작성했나요?**
    - 프로젝트의 진행 개요 및 프로세스, 결과 분석이 잘 정리되어 있었고, 한계 포인트를 별도로 정리하고 이후 진행 방향에 대해서 잘 정리되어 있었습니다.  
        <img width="871" height="1097" alt="image" src="https://github.com/user-attachments/assets/50c3d41b-6e16-4770-b4af-430ac40b71c9" />

        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - advanced_rag_klue 함수로 문서검색 > Reranking > 축소 > LLM 답변 생성 과정을 모듈화하여 이후 평가단계에서 다시 호출하여 사용함으로써 직관적이고 간결하게 유지하였습니다.  
        <img width="767" height="387" alt="image" src="https://github.com/user-attachments/assets/8b51dbb0-137e-488f-b84c-2fa422199526" />




# 회고(참고 링크 및 코드 개선)
```
# 리뷰어의 회고를 작성합니다.
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
단순히 점수만 비교하는 데 그치지 않고, "왜 이런 결과가 나왔는지" 뉴스 기사의 특성과 연결하여 깊이 있게 분석한 점이 매우 인상깊었습니다.  
