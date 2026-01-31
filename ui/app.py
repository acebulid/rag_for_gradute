import streamlit as st
import asyncio
from core.service_caller import init_service, process_user_query, close_service

# ---------------------- 页面基础配置（先执行，不变） ----------------------
st.set_page_config(
    page_title="RAG查询系统",
    layout="wide"  # 宽布局，支撑分栏效果
)

# ---------------------- 初始化全局缓存（历史对话+服务状态） ----------------------
if "service_initialized" not in st.session_state:
    st.session_state["service_initialized"] = False
if "chat_history" not in st.session_state:
    # 历史对话格式：[(查询内容, 回复结果, 查询状态), ...]
    st.session_state["chat_history"] = []

# ---------------------- 核心：1:3 分栏布局（左侧历史+右侧对话） ----------------------
# 定义左右两栏，比例 1:3（对应 1/4 和 3/4 宽度）
left_col, right_col = st.columns([1, 3])

# ========== 左侧栏：1/4 宽度 - 历史对话栏 ==========
with left_col:
    st.title("历史对话")
    st.divider()
    
    # 清空历史对话按钮
    if st.button("清空全部历史", type="secondary"):
        st.session_state["chat_history"] = []
        # 刷新页面（让清空效果立即生效）
        st.rerun()
    
    # 展示历史对话（倒序，最新的在最上方）
    if st.session_state["chat_history"]:
        for idx, (query, response, is_success) in enumerate(reversed(st.session_state["chat_history"])):
            # 历史对话卡片（带状态标识）
            with st.expander(f"查询 {len(st.session_state['chat_history']) - idx}" if is_success else f"❌ 查询 {len(st.session_state['chat_history']) - idx}", expanded=False):
                st.markdown("**你的查询：**")
                st.write(query)
                st.markdown("**系统回复：**")
                st.write(response)
                
                # 回显按钮（点击后将历史查询填入右侧输入框）
                if st.button(f"重新查询", key=f"requery_{idx}"):
                    # 将历史查询存入session_state，供右侧输入框读取
                    st.session_state["current_query"] = query
                    st.rerun()
    else:
        st.info("暂无历史对话，开始你的第一次查询吧～")

# ========== 右侧栏：3/4 宽度 - 核心对话区域 ==========
with right_col:
    st.title("首都师范大学RAG查询系统")
    st.divider()
    
    # 1. 服务初始化（仅执行一次）
    if not st.session_state["service_initialized"]:
        try:
            asyncio.run(init_service())
            st.session_state["service_initialized"] = True
            st.success("服务初始化成功，可开始查询！")
        except Exception as e:
            st.session_state["service_initialized"] = False
            st.error(f"服务初始化失败：{str(e)}")
    
    # 2. 输入区域（支持回显历史查询）
    st.subheader("输入查询内容")
    # 初始化current_query，避免KeyError
    if "current_query" not in st.session_state:
        st.session_state["current_query"] = ""
    # 文本输入框（读取session_state中的current_query，实现回显）
    user_query = st.text_area(
        label="请输入你的查询（例如：首都师范大学的校门在哪里？）",
        height=100,
        placeholder="在这里输入查询内容，点击下方按钮提交...",
        disabled=not st.session_state["service_initialized"],
        value=st.session_state["current_query"]  # 回显历史查询
    )
    
    # 3. 提交按钮（居中显示）
    col1, col2, col3 = st.columns(3)
    with col2:
        submit_btn = st.button(
            label="提交查询",
            type="primary",
            disabled=not (st.session_state["service_initialized"] and user_query.strip())
        )
    
    # 4. 结果展示区域
    st.divider()
    st.subheader("查询结果")
    result_container = st.empty()
    
    # 5. 提交按钮点击事件（处理查询+保存历史）
    if submit_btn and user_query.strip():
        result_container.info(" 正在处理查询，请稍候...")
        
        try:
            # 调用核心服务处理查询
            query_result = asyncio.run(
                process_user_query(
                    query_type="text",
                    query_content=user_query.strip()
                )
            )
            
            # 处理查询结果
            if query_result["success"]:
                result_container.success("查询成功！")
                response_content = query_result["polished_response"]
                st.markdown(f"### 回复内容\n{response_content}")
                
                # 保存到历史对话（查询内容、回复结果、成功状态）
                st.session_state["chat_history"].append(
                    (user_query.strip(), response_content, True)
                )
            else:
                result_container.error(f"查询失败：{query_result['error']}")
                response_content = query_result["error"]
                
                # 保存到历史对话（查询内容、错误信息、失败状态）
                st.session_state["chat_history"].append(
                    (user_query.strip(), response_content, False)
                )
            
            # 清空当前输入框的回显标记
            st.session_state["current_query"] = ""
            
        except Exception as e:
            error_msg = f" 处理查询时发生异常：{str(e)}"
            result_container.error(error_msg)
            
            # 保存异常到历史对话
            st.session_state["chat_history"].append(
                (user_query.strip(), error_msg, False)
            )
    
    # 6. 系统信息折叠栏
    st.divider()
    with st.expander("🔧 系统信息", expanded=False):
        st.write("• 服务状态：已初始化" if st.session_state["service_initialized"] else "• 服务状态：未初始化")
        st.write(f"• 历史对话条数：{len(st.session_state['chat_history'])}")
        st.write("• 提示：关闭页面后，服务会自动释放资源")