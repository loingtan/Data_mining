import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Load mô hình SVR đã huấn luyện
@st.cache_resource
def load_model():
    return joblib.load("svr.pkl")

model = load_model()

# Danh sách các cột đặc trưng dùng để dự đoán
FEATURE_COLS = [
    'video_completion', 'problem_completion', 'alpha',
    'num_videos', 'num_problems', 'num_teacher', 'num_school',
    'field_encoded', 'prerequisites_encoded', 'num_exercises',
    'num_students', 'total_default_video_time', 'total_comments',
    'total_replies', 'avg_comments_per_student', 'avg_replies_per_student',
    'total_problem_attempts', 'avg_problem_attempts_per_student',
    'course_total_completion_rate', 'course_avg_completion_rate',
    'total_video_watch_time', 'avg_video_watch_time_per_student',
    'problem_iscorrect_ratio', 'problem_attempts_ratio',
    'problem_score_ratio', 'problem_lang_ratio', 'problem_option_ratio',
    'problem_type_ratio', 'user_total_video_watch_time',
    'user_avg_video_watch_time', 'video_watched'
]

# Load dữ liệu
@st.cache_data
def load_data():
    return pd.read_csv("dataset_final.csv")  

df = load_data()

df['predicted_completion'] = model.predict(df[FEATURE_COLS].fillna(0))

# Sidebar chọn chế độ
mode = st.sidebar.radio("Chế độ xem", ["📘 Giáo viên - Theo Khóa học", "🎓 Giáo viên - Theo Học sinh", "🙋‍♂️ Học sinh (Cá nhân)"])

# ========== CHẾ ĐỘ: GIÁO VIÊN - THEO KHÓA HỌC ==========
if mode == "📘 Giáo viên - Theo Khóa học":
    st.title("📊 Thống kê theo Khóa học")

    course_ids = df['course_id'].unique()
    selected_course = st.selectbox("Chọn khóa học", course_ids)

    course_data = df[df['course_id'] == selected_course]

    st.subheader("📝 Thông tin khóa học")
    st.write(f"**Tổng số học sinh:** {int(course_data['num_students'].values[0])}")
    st.write(f"**Số lượng video:** {int(course_data['num_videos'].values[0])}")
    st.write(f"**Số lượng bài tập:** {int(course_data['num_problems'].values[0])}")
    st.write(f"**Số giáo viên:** {int(course_data['num_teacher'].values[0])}")
    st.write(f"**Số trường tham gia:** {int(course_data['num_school'].values[0])}")

    st.subheader("📈 Tương tác học sinh")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⏱ Thời gian xem video TB", f"{course_data['avg_video_watch_time_per_student'].values[0]:.1f} giây")
        st.metric("💬 Bình luận TB", f"{course_data['avg_comments_per_student'].values[0]:.2f}")
        st.metric("📑 Số lần làm bài TB", f"{course_data['avg_problem_attempts_per_student'].values[0]:.2f}")
    with col2:
        st.metric("📬 Trả lời TB", f"{course_data['avg_replies_per_student'].values[0]:.2f}")
        st.metric("🎯 Tỉ lệ đúng", f"{course_data['problem_iscorrect_ratio'].values[0]*100:.1f}%")
        st.metric("📊 Tỉ lệ hoàn thành TB", f"{course_data['course_avg_completion_rate'].values[0]*100:.1f}%")

    st.subheader("👨‍🎓 Danh sách học sinh")
    student_list = course_data[['user_id', 'video_completion', 'problem_completion', 'completion', 'predicted_completion']]
    student_list.columns = ['User ID', 'Video %', 'Problem %', 'Overall Completion', 'Predicted Completion']
    student_list[['Video %', 'Problem %', 'Overall Completion', 'Predicted Completion']] *= 100
    st.dataframe(student_list.style.format({
        "Video %": "{:.1f}",
        "Problem %": "{:.1f}",
        "Overall Completion": "{:.1f}",
        "Predicted Completion": "{:.1f}"
    }))
    
    st.subheader("📊 Biểu đồ phân tích học sinh")

    # Vẽ biểu đồ Completion
    fig1, ax1 = plt.subplots()
    sns.histplot(student_list['Overall Completion'], bins=10, kde=True, ax=ax1)
    ax1.set_title("Phân phối tỉ lệ hoàn thành (%)")
    ax1.set_xlabel("Completion (%)")
    st.pyplot(fig1)

    # Vẽ biểu đồ Video Completion
    fig2, ax2 = plt.subplots()
    sns.histplot(student_list['Video %'], bins=10, color='orange', kde=True, ax=ax2)
    ax2.set_title("Phân phối tỉ lệ hoàn thành video (%)")
    ax2.set_xlabel("Video Completion (%)")
    st.pyplot(fig2)

    # Vẽ biểu đồ Problem Completion
    fig3, ax3 = plt.subplots()
    sns.histplot(student_list['Problem %'], bins=10, color='green', kde=True, ax=ax3)
    ax3.set_title("Phân phối tỉ lệ hoàn thành bài tập (%)")
    ax3.set_xlabel("Problem Completion (%)")
    st.pyplot(fig3)

# ========== CHẾ ĐỘ: GIÁO VIÊN - THEO HỌC SINH ==========
elif mode == "🎓 Giáo viên - Theo Học sinh":
    st.title("🎓 Thống kê học sinh")

    user_ids = df['user_id'].unique()
    selected_user = st.selectbox("Chọn học sinh", user_ids)

    user_data = df[df['user_id'] == selected_user]

    st.write(f"**Thuộc khóa học:** {user_data['course_id'].values[0]}")

    st.subheader("📚 Tiến độ học tập")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📺 Video hoàn thành", f"{user_data['video_completion'].values[0]*100:.1f}%")
        st.metric("📝 Bài tập hoàn thành", f"{user_data['problem_completion'].values[0]*100:.1f}%")
        st.metric("🎯 Điểm completion (α)", f"{user_data['alpha'].values[0]:.2f}")
    with col2:
        st.metric("🏁 Tỉ lệ hoàn thành khóa học", f"{user_data['completion'].values[0]*100:.1f}%")
        st.metric("📈 Tỉ lệ đúng bài tập", f"{user_data['problem_iscorrect_ratio'].values[0]*100:.1f}%")
        st.metric("🔮 Dự đoán (SVR)", f"{user_data['predicted_completion'].values[0]*100:.1f}%")

    st.subheader("📺 Tương tác với video")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⏱ Tổng thời gian xem", f"{user_data['user_total_video_watch_time'].values[0]:.1f} giây")
        st.metric("📦 Video đã xem", f"{int(user_data['video_watched'].values[0])}")
    with col2:
        st.metric("⏱ TB mỗi video", f"{user_data['user_avg_video_watch_time'].values[0]:.1f} giây")

    st.subheader("💬 Tương tác xã hội")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 Tổng bình luận", f"{user_data['total_comments'].values[0]:.0f}")
    with col2:
        st.metric("📬 Tổng trả lời", f"{user_data['total_replies'].values[0]:.0f}")
        
    st.subheader("📊 Tổng quan toàn bộ học sinh")

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x='completion', color='skyblue', ax=ax4)
    ax4.set_title("Boxplot: Tỉ lệ hoàn thành của tất cả học sinh")
    ax4.set_xlabel("Completion")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(8, 4))
    sns.histplot(df['user_total_video_watch_time'], bins=20, color='purple', kde=True, ax=ax5)
    ax5.set_title("Thời gian xem video (tất cả học sinh)")
    ax5.set_xlabel("Seconds")
    st.pyplot(fig5)

# ========== CHẾ ĐỘ: HỌC SINH (CÁ NHÂN) ==========
else:
    st.title("🙋‍♂️ Giao diện học sinh")

    # Chọn học sinh
    user_ids = df['user_id'].unique()
    selected_user = st.selectbox("Chọn học sinh (xem riêng tư)", user_ids)

    # Lấy dữ liệu của học sinh
    user_data = df[df['user_id'] == selected_user]

    # Hiển thị thông tin khóa học và học sinh
    st.write(f"**Khóa học bạn đang học:** {user_data['course_id'].values[0]}")
    st.write(f"**Họ và tên học sinh:** {user_data['user_id'].values[0]}")  # Thay bằng thông tin tên học sinh nếu có

    st.subheader("📚 Tiến độ học tập cá nhân")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📺 Video hoàn thành", f"{user_data['video_completion'].values[0]*100:.1f}%")
        st.metric("📝 Bài tập hoàn thành", f"{user_data['problem_completion'].values[0]*100:.1f}%")
        st.metric("🎯 Điểm completion (α)", f"{user_data['alpha'].values[0]:.2f}")
    with col2:
        st.metric("🏁 Tỉ lệ hoàn thành khóa học", f"{user_data['completion'].values[0]*100:.1f}%")
        st.metric("📈 Tỉ lệ đúng bài tập", f"{user_data['problem_iscorrect_ratio'].values[0]*100:.1f}%")
        st.metric("🔮 Dự đoán (SVR)", f"{user_data['predicted_completion'].values[0]*100:.1f}%")

    # Hiển thị thông tin khóa học chi tiết
    course_id = user_data['course_id'].values[0]
    course_data = df[df['course_id'] == course_id]

    st.subheader("📝 Thông tin khóa học")
    st.write(f"**Tổng số học sinh trong khóa học:** {int(course_data['num_students'].values[0])}")
    st.write(f"**Số lượng video trong khóa học:** {int(course_data['num_videos'].values[0])}")
    st.write(f"**Số lượng bài tập trong khóa học:** {int(course_data['num_problems'].values[0])}")
    st.write(f"**Số giáo viên trong khóa học:** {int(course_data['num_teacher'].values[0])}")
    st.write(f"**Số trường tham gia khóa học:** {int(course_data['num_school'].values[0])}")

    st.subheader("📊 Tương tác học sinh trong khóa học")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⏱ Thời gian xem video TB khóa học", f"{course_data['avg_video_watch_time_per_student'].values[0]:.1f} giây")
        st.metric("💬 Bình luận TB khóa học", f"{course_data['avg_comments_per_student'].values[0]:.2f}")
    with col2:
        st.metric("📑 Số lần làm bài TB khóa học", f"{course_data['avg_problem_attempts_per_student'].values[0]:.2f}")
        st.metric("📬 Trả lời TB khóa học", f"{course_data['avg_replies_per_student'].values[0]:.2f}")
    
    # Vẽ biểu đồ về khóa học
    st.subheader("📊 Biểu đồ phân tích khóa học")

    # Biểu đồ tỷ lệ hoàn thành khóa học
    fig1, ax1 = plt.subplots()
    sns.histplot(course_data['course_avg_completion_rate'], bins=10, kde=True, ax=ax1)
    ax1.set_title("Phân phối tỉ lệ hoàn thành khóa học (%)")
    ax1.set_xlabel("Completion (%)")
    st.pyplot(fig1)

    # Biểu đồ thời gian xem video khóa học
    fig2, ax2 = plt.subplots()
    sns.histplot(course_data['avg_video_watch_time_per_student'], bins=10, color='orange', kde=True, ax=ax2)
    ax2.set_title("Phân phối thời gian xem video khóa học (giây)")
    ax2.set_xlabel("Video Watch Time (s)")
    st.pyplot(fig2)
