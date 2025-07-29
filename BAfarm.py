import streamlit as st
import numpy as np
import pulp
import pandas as pd

st.title("노말 파밍 최적화 계산기")

# Event Ratio Selector
event_ratio = st.radio(
    "이벤트 여부:",
    [1, 2, 3],
    index=0,
    format_func=lambda x: f"{x}배",
    horizontal=True
)
# Equipment Tier Selector
tier = st.selectbox("장비 티어:", list(range(1, 11)))

# Target Input
target = []
item_labels = ['모자', '장갑', '신발', '가방', '배지', '헤어핀', '부적', '시계', '목걸이']
st.write("## 목표 아이템 수 입력")
cols = st.columns(9)
for i, col in enumerate(cols):
    val = col.number_input(f"{item_labels[i]}", min_value=0, value=0)
    target.append(val)
target = np.array(target)

V_type1 = event_ratio * np.array([
    [0.4,   0.3,   0.3,   0,     0,     0,     0,     0,     0],
    [0.3,   0.4,   0,     0.3,   0,     0,     0,     0,     0],
    [0.3,   0.3,   0.4,   0,     0,     0,     0,     0,     0],
    [0,     0,     0.3,   0.4,   0,     0.3,   0,     0,     0],
    [0.3,   0,     0,     0.3,   0.4,   0,     0,     0,     0],
    [0,     0,     0,     0,     0.3,   0.4,   0,     0.3,   0],
    [0,     0,     0,     0,     0,     0.4,   0,     0.3,   0.3],
    [0,     0,     0,     0,     0,     0.3,   0.4,   0.3,   0],
    [0,     0,     0,     0,     0,     0,     0.3,   0.4,   0.3],
    [0,     0,     0,     0,     0.3,   0,     0.3,   0,     0.4]
])
V_type2 = event_ratio * np.array([
    [1.2,   0.9,   0.9,   0,     0,     0,     0,     0,     0],
    [0.9,   1.2,   0,     0.9,   0,     0,     0,     0,     0],
    [0.9,   0.9,   1.2,   0,     0,     0,     0,     0,     0],
    [0,     0,     0.9,   1.2,   0,     0.9,   0,     0,     0],
    [0.9,   0,     0,     0.9,   1.2,   0,     0,     0,     0],
    [0,     0,     0,     0,     0.9,   1.2,   0,     0.9,   0],
    [0,     0,     0,     0,     0,     1.2,   0,     0.9,   0.9],
    [0,     0,     0,     0,     0,     0.9,   1.2,   0.9,   0],
    [0,     0,     0,     0,     0,     0,     0.9,   1.2,   0.9],
    [0,     0,     0,     0,     0.9,   0,     0.9,   0,     1.2]
])
V_type3 = event_ratio * np.array([
    [1,     0.75,  0.75,  0,     0,     0,     0,     0,     0],
    [0.75,  1,     0,     0.75,  0,     0,     0,     0,     0],
    [0.75,  0.75,  1,     0,     0,     0,     0,     0,     0],
    [0,     0,     0.75,  1,     0,     0.75,  0,     0,     0],
    [0.75,  0,     0,     0.75,  1,     0,     0,     0,     0],
    [0,     0,     0,     0,     0.75,  1,     0,     0.75,  0],
    [0,     0,     0,     0,     0,     1,     0,     0.75,  0.75],
    [0,     0,     0,     0,     0,     0.75,  1,     0.75,  0],
    [0,     0,     0,     0,     0,     0,     0.75,  1,     0.75],
    [0,     0,     0,     0,     0.75,  0,     0.75,  0,     1]
])
V_type4 = event_ratio * np.array([
    [0.8,   0.6,   0.6,   0,     0,     0,     0,     0,     0],
    [0.6,   0.8,   0,     0.6,   0,     0,     0,     0,     0],
    [0.6,   0.6,   0.8,   0,     0,     0,     0,     0,     0],
    [0,     0,     0.6,   0.8,   0,     0.6,   0,     0,     0],
    [0.6,   0,     0,     0.6,   0.8,   0,     0,     0,     0],
    [0,     0,     0,     0,     0.6,   0.8,   0,     0.6,   0],
    [0,     0,     0,     0,     0,     0.8,   0,     0.6,   0.6],
    [0,     0,     0,     0,     0,     0.6,   0.8,   0.6,   0],
    [0,     0,     0,     0,     0,     0,     0.6,   0.8,   0.6],
    [0,     0,     0,     0,     0.6,   0,     0.6,   0,     0.8]
])
V_type5 = event_ratio * np.array([
    [0.6,   0.45,  0.45,  0,     0,     0,     0,     0,     0],
    [0.45,  0.6,   0,     0,     0.45,  0,     0,     0,     0],
    [0.45,  0,     0.6,   0,     0,     0.45,  0,     0,     0],
    [0,     0,     0.45,  0.45,  0,     0.6,   0,     0,     0],
    [0.45,  0.45,  0,     0,     0.6,   0,     0,     0,     0],
    [0,     0,     0,     0,     0,     0,     0.6,   0.45,  0.45],
    [0,     0,     0,     0,     0,     0.6,   0.45,  0.45,  0],
    [0,     0,     0,     0.6,   0,     0,     0,     0.45,  0.45],
    [0,     0,     0,     0,     0.45,  0,     0.45,  0.6,   0],
    [0,     0,     0,     0.45,  0,     0.45,  0,     0,     0.6]
])
V_type6 = event_ratio * np.array([
    [0.344, 0.258, 0.258, 0,     0,     0,     0,     0,     0],
    [0.258, 0.344, 0,     0,     0.258, 0,     0,     0,     0],
    [0.258, 0,     0.344, 0,     0,     0.258, 0,     0,     0],
    [0,     0,     0.258, 0.258, 0,     0.344, 0,     0,     0],
    [0.258, 0.258, 0,     0,     0.344, 0,     0,     0,     0],
    [0,     0,     0,     0,     0,     0,     0.344, 0.258, 0.258],
    [0,     0,     0,     0,     0,     0.344, 0.258, 0.258, 0],
    [0,     0,     0,     0.344, 0,     0,     0,     0.258, 0.258],
    [0,     0,     0,     0,     0.258, 0,     0.258, 0.344, 0],
    [0,     0,     0,     0.258, 0,     0.258, 0,     0,     0.344]
])
V_type7 = event_ratio * np.array([
    [0.344, 0.258, 0.258, 0,     0,     0,     0,     0,     0],
    [0.258, 0.344, 0,     0,     0.258, 0,     0,     0,     0],
    [0.258, 0,     0.344, 0,     0,     0.258, 0,     0,     0],
    [0,     0,     0,     0.258, 0.258, 0.344, 0,     0,     0],
    [0.258, 0.258, 0,     0,     0.344, 0,     0,     0,     0],
    [0,     0,     0,     0,     0,     0,     0.344, 0.258, 0.258],
    [0,     0,     0.258, 0,     0,     0.344, 0,     0.258, 0],
    [0,     0,     0,     0.344, 0,     0,     0.258, 0.258, 0],
    [0,     0.258, 0,     0,     0.258, 0,     0,     0.344, 0],
    [0,     0,     0,     0.258, 0,     0.258, 0,     0,     0.344]
])
V_type8 = event_ratio * np.array([
    [0.344, 0.258, 0.258, 0,     0,     0,     0,     0,     0],
    [0.258, 0.344, 0,     0,     0.258, 0,     0,     0,     0],
    [0.258, 0,     0.344, 0,     0,     0.258, 0,     0,     0],
    [0,     0,     0,     0.258, 0.258, 0.344, 0,     0,     0],
    [0.258, 0.258, 0,     0,     0.344, 0,     0,     0,     0],
    [0,     0,     0,     0,     0,     0,     0,     0,     0],
    [0,     0,     0,     0,     0,     0,     0,     0,     0],
    [0,     0,     0,     0,     0,     0,     0,     0,     0],
    [0,     0,     0,     0,     0,     0,     0,     0,     0],
    [0,     0,     0,     0,     0,     0,     0,     0,     0]
])
V_dic={1: V_type1, 2: V_type2, 3: V_type3, 4: V_type4, 5: V_type5, 6: V_type6, 7: V_type6, 8: V_type6, 9: V_type7, 10: V_type8}


label_numeric = {'하': 0, '중': 1, '상': 2}
priority_weights = {}
priority_weights_initVal = {'모자': '상', '장갑': '상', '신발': '중', '가방': '중', '배지': '중', '헤어핀': '상', '부적': '하', '시계': '상', '목걸이': '중'}

st.markdown("## 우선순위 설정")

cols = st.columns(3)
for idx, name in enumerate(item_labels):
    with cols[idx % 3]:
        val = st.select_slider(
            f"{name} 등급", 
            options=['하','중','상'],
            value=priority_weights_initVal[name],
            label_visibility="visible"
        )
        priority_weights[name] = val

# Row Activation Switches
st.write("## 소탕 가능 지역")

low_labels = [f"Low - {i+1} 지역" for i in range(5)]
high_labels = [f"High - {i+1} 지역" for i in range(5)]
row_active = []
low_cols = st.columns(5)
for i in range(5):
    row_active.append(low_cols[i].checkbox(low_labels[i], value=True, key=f"chk_low_{i+1}"))
high_cols = st.columns(5)
for i in range(5):
    row_active.append(high_cols[i].checkbox(high_labels[i], value=True, key=f"chk_high_{i+1}"))
    
V = V_dic[tier].copy()
for i in range(10):
    if not row_active[i]:
        V[i] = [0,     0,     0,     0,     0,     0,     0,     0,     0]

# Solve Button
if st.button("계산 시작"):
    
    if np.all(target == 0):
        st.warning("모든 목표가 0입니다. 최소 하나 이상의 값을 입력해주세요.")
        st.stop()
        
    # Phase 1: minimize total repetitions
    prob1 = pulp.LpProblem("Minimize_total", pulp.LpMinimize)
    n_vars = [pulp.LpVariable(f"n{i+1}", lowBound=0, cat='Integer') for i in range(10)]
    total = pulp.lpSum(n_vars)
    prob1 += total
    for j in range(9):
        prob1 += pulp.lpSum(n_vars[i] * V[i, j] for i in range(10)) >= target[j]
    prob1.solve()

    if pulp.LpStatus[prob1.status] != 'Optimal':
        st.error("해결할 수 없습니다. 조건을 다시 확인해 주세요.")
    else:
        min_total = pulp.value(total)

        # Phase 2: maximize priority-weighted gain under min_total constraint
        prob2 = pulp.LpProblem("Maximize_priority", pulp.LpMaximize)
        n_vars2 = [pulp.LpVariable(f"n{i+1}", lowBound=0, cat='Integer') for i in range(10)]
        prob2 += pulp.lpSum(
            n_vars2[i] * sum(label_numeric[priority_weights[item_labels[j]]] * V[i, j] for j in range(9))
            for i in range(10)
        )

        for j in range(9):
            prob2 += pulp.lpSum(n_vars2[i] * V[i, j] for i in range(10)) >= target[j]
        prob2 += pulp.lpSum(n_vars2) == min_total
        prob2.solve()

        # 지역 결과 정리

        region_names = []
        counts = []
        region_group = 3 * tier - 2

        for i, var in enumerate(n_vars2):
            count = int(var.varValue)
            group_idx = i % 5 + 1
            part_idx = i // 5 + 1
            region = f"{region_group + part_idx - 1} - {group_idx}"
            region_names.append(region)
            counts.append(f"{count}회")

        # region data
        region_table =  pd.DataFrame({
            "지역": region_names,
            "소탕 횟수": counts
        })
        
        # Display
        st.markdown("## 결과")
        st.dataframe(region_table.set_index("지역").T)

        total_ap = int(sum(var.varValue for var in n_vars2)) * 10  # If AP per try is ap_cost
        st.text(f"총 사용 AP:  {total_ap}")

        # 최종 장비 기대값 및 차이
        st.write("## 장비 기대값 비교")

        expected = np.zeros(9)
        for i in range(10):
            expected += int(n_vars2[i].varValue) * V[i]

        expected_floor = np.floor(expected).astype(int)
        remains = np.maximum(expected_floor - target, 0)

        comparison_table = pd.DataFrame({
            "장비 이름": item_labels,
            "목표 수량": target,
            "기대 수량": expected_floor,
            "초과 수량": remains
        })

        st.dataframe(comparison_table.set_index("장비 이름").T)


# --- Custom CSS ---
hide_elements = """
        <style>
            div[data-testid="stSliderTickBarMin"],
            div[data-testid="stSliderTickBarMax"] {
                display: none;
            }
            .stSlider {
                padding-bottom: 0.5rem;
            }
            .stSlider label {
                display: block;
                text-align: center;
            }
        </style>
"""

st.markdown(hide_elements, unsafe_allow_html=True)
