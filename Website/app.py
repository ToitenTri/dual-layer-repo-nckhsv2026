import os
import time
import uuid
import cv2
from flask import Flask, render_template, request, flash, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename

from config import Config
from database import Database
from core.crypto_manager import CryptoManager
from core.watermark import embed_watermark, extract_watermark
from core.adversarialnoise import add_adversarial_noise

app = Flask(__name__)
app.config.from_object(Config)

Config.init_Folders()
db = Database()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def user_portal():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            flash('Lỗi: Không tìm thấy file!', 'error')
            return redirect(request.url)

        file = request.files['image_file']
        author_id = request.form.get('author_id', 'Anonymous').strip()
        secret_text = request.form.get('watermark_text', '').strip()

        if file.filename == '' or secret_text == '':
            flash('Lỗi: Vui lòng điền đầy đủ thông tin!', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                start_time = time.time()
                crypto_pkg = CryptoManager.generate_watermark_package(secret_text, author_id)
                wm_id = crypto_pkg["watermark_id"]
                CryptoManager.save_key_file(crypto_pkg)
                key_filename = f"{wm_id}.json"

                original_filename = secure_filename(file.filename)
                ext = original_filename.rsplit('.', 1)[1].lower()
                saved_input_name = f"in_{wm_id}.{ext}"
                output_name = f"out_{wm_id}.png"

                input_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_input_name)
                output_path = os.path.join(app.config['RESULT_FOLDER'], output_name)
                file.save(input_path)

                img_matrix = cv2.imread(input_path)
                if img_matrix is None:
                    os.remove(input_path)
                    flash('Lỗi: File ảnh bị hỏng!', 'error')
                    return redirect(request.url)

                watermarked_img = embed_watermark(img_matrix, crypto_pkg["embedded_bits_str"])
                final_protected_img = add_adversarial_noise(watermarked_img, epsilon=0.03)
                cv2.imwrite(output_path, final_protected_img)

                process_time = time.time() - start_time
                db.save_full_transaction(crypto_pkg, key_filename, original_filename, saved_input_name, output_name,
                                         process_time)

                flash(f'Thành công! Mã định danh bản quyền: {wm_id}', 'success')
                return render_template('user_portal.html', wm_id=wm_id, input_image=saved_input_name,
                                       output_image=output_name, key_filename=key_filename,
                                       process_time=round(process_time, 3))
            except Exception as e:
                flash(f'Lỗi hệ thống: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Lỗi định dạng file!', 'error')
            return redirect(request.url)
    return render_template('user_portal.html')


@app.route('/download-key/<filename>')
def download_key(filename):
    return send_from_directory(app.config['KEY_STORAGE_FOLDER'], filename, as_attachment=True)


@app.route('/manager/login', methods=['GET', 'POST'])
def manager_login():
    if request.method == 'POST':
        if request.form.get('username') == Config.ADMIN_USERNAME and request.form.get(
                'password') == Config.ADMIN_PASSWORD:
            session['is_manager'] = True
            return redirect(url_for('manager_dashboard'))
        flash('Sai tài khoản hoặc mật khẩu!', 'error')
    return render_template('manager_login.html')


@app.route('/manager/logout')
def manager_logout():
    session.pop('is_manager', None)
    return redirect(url_for('user_portal'))


@app.route('/manager', methods=['GET', 'POST'])
def manager_dashboard():
    if not session.get('is_manager'):
        return redirect(url_for('manager_login'))

    verification_result = None
    extracted_bits = None
    if request.method == 'POST' and 'verify_image' in request.files:
        v_file = request.files['verify_image']
        if v_file and allowed_file(v_file.filename):
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_verify.png")
            v_file.save(temp_path)
            v_matrix = cv2.imread(temp_path)
            if v_matrix is not None:
                extracted_bits = extract_watermark(v_matrix)
                verification_result = db.verify_extracted_bits(extracted_bits)
                os.remove(temp_path)

    all_records = db.get_all_joined_records()
    return render_template('manager_dashboard.html', records=all_records, verification_result=verification_result,
                           extracted_bits=extracted_bits)


@app.route('/manager/delete/<wm_id>', methods=['POST'])
def manager_delete(wm_id):
    if not session.get('is_manager'):
        return redirect(url_for('manager_login'))
    db.delete_record_by_wmid(wm_id)
    return redirect(url_for('manager_dashboard'))


@app.errorhandler(413)
def request_entity_too_large(error):
    flash('Lỗi: Dung lượng ảnh vượt quá 16MB!', 'error')
    return redirect(url_for('user_portal'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)