import os
import uuid
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, Response # <-- 导入 Response
from werkzeug.utils import secure_filename
from processing_pipeline import run_pipeline # 导入重构后的处理函数
import pprint # <-- 导入 pprint 用于调试打印

# ==============================================================================
# 自定义 JSON 编码器 (Custom JSON Encoder)
# ==============================================================================
class CustomJSONEncoder(json.JSONEncoder):
    """
    创建一个自定义的JSON编码器来处理Numpy的数据类型。
    当jsonify遇到它不认识的数据类型时（比如np.int64），它会调用这个类的default方法。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomJSONEncoder, self).default(obj)

# ==============================================================================
# Flask 应用配置
# ==============================================================================
app = Flask(__name__, template_folder='templates')
# 注意：尽管我们将手动调用它，但保留此行作为备用并无坏处
app.json_encoder = CustomJSONEncoder 

# 配置上传文件夹和允许的文件扩展名
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 确保上传和输出目录存在
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否被允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================================================================
# 路由定义
# ==============================================================================
@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    """处理上传的CSV文件"""
    if 'csv-file' not in request.files:
        return jsonify({'error': '请求中没有文件部分'}), 400
    
    file = request.files['csv-file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
        
    if file and allowed_file(file.filename):
        # 1. 为本次处理创建一个唯一的会话目录
        session_id = str(uuid.uuid4())
        session_upload_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
        session_output_dir = Path(app.config['OUTPUT_FOLDER']) / session_id
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        session_output_dir.mkdir(parents=True, exist_ok=True)

        # 2. 【关键改动】保存上传的文件时，直接使用原始文件名
        # 我们不再使用 secure_filename，因为它会错误地处理中文字符
        filename = file.filename
        input_filepath = session_upload_dir / filename
        file.save(input_filepath)

        print(f"[{session_id}] 文件已保存到: {input_filepath}")

        try:
            # 3. 调用处理流水线
            print(f"[{session_id}] 开始处理...")
            results = run_pipeline(str(input_filepath), str(session_output_dir))
            print(f"[{session_id}] 处理完成。")
            
            # --- 调试代码可以保留或移除 ---
            print("\n--- 准备返回给前端的数据 (调试信息) ---")
            pprint.pprint(results)
            print("\n--- 数据类型检查 ---")
            for key, value in results.items():
                print(f"键: '{key}', 类型: {type(value)}")
            print("----------------------------------------\n")
            
            # 4. 返回处理结果
            final_report_path = Path(results['final_summary_report_path'])
            results['download_url'] = f'/download/{session_id}/{final_report_path.name}'
            
            # 手动使用自定义编码器将结果序列化为JSON字符串
            json_response = json.dumps(results, cls=CustomJSONEncoder)
            
            # 创建一个Flask Response对象，并设置正确的MIME类型
            return Response(json_response, mimetype='application/json')

        except Exception as e:
            print(f"[错误] 处理文件时发生错误: {e}")
            # 对于内部服务器错误，我们也返回一个JSON响应
            error_response = json.dumps({'error': f'处理过程中发生错误: {str(e)}'})
            return Response(error_response, status=500, mimetype='application/json')
            
    return jsonify({'error': '文件类型不允许'}), 400

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """提供结果文件的下载"""
    directory = Path(app.config['OUTPUT_FOLDER']) / session_id
    return send_from_directory(directory, filename, as_attachment=True)


# ==============================================================================
# 启动应用
# ==============================================================================
if __name__ == '__main__':
    # 在生产环境中，应使用专业的WSGI服务器如Gunicorn或uWSGI
    app.run(debug=True, port=5001)

