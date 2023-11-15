from flask import Flask, jsonify
import pickle
import numpy as np
from waitress import serve

from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

import re


test_data = ['MessageDigest instance = MessageDigest.getInstance("sha-1");',
             'Log.e("Java/LocalDNSServer", "Failed to create server socket");',
             'builder.addAddress("10.1.10.1", 32);',
             'if (intValue != 0 && !TextUtils.isEmpty("127.0.0.1")) {',
             'import java.util.Random;',
             'public static final String OWM_API_KEY = "82eff2c845841c89c837d4e125613d83";',
             'System.out.println(zone);',
             'import java.security.MessageDigest;',
             'String name = "MyApp";',
             'System.out.println("echo <text>\n\nprints the text\n");']

CWE_Desc = {
    "CWE-89": "The software constructs all or part of an SQL command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command when it is sent to a downstream component.",
    "CWE-200": "The product exposes sensitive information to an actor that is not explicitly authorized to have access to that information.",
    "CWE-276": "During installation, installed file permissions are set to allow anyone to modify those files.",
    "CWE-312": "The application stores sensitive information in cleartext within a resource that might be accessible to another control sphere.",
    "CWE-532": "Information written to log files can be of a sensitive nature and give valuable guidance to an attacker or expose sensitive user information.",
    "CWE-676": "The program invokes a potentially dangerous function that could introduce a vulnerability if it is used incorrectly, but the function can also be used safely.",
    "CWE-749": "The software provides an Applications Programming Interface (API) or similar interface for interaction with external actors, but the interface includes a dangerous method or function that is not properly restricted.",
    "CWE-921": "The software stores sensitive information in a file system or device that does not have built-in access control.",
    "CWE-925": "The Android application uses a Broadcast Receiver that receives an Intent but does not properly verify that the Intent came from an authorized source.",
    "CWE-939": "The software uses a handler for a custom URL scheme, but it does not properly restrict which actors can invoke the handler using the scheme.",
    "other": "Some other vulnerability",
    "": "Non-vulnerable code"
}

CWE_Links = {
    "CWE-89": "https://cwe.mitre.org/data/definitions/89.html",
    "CWE-200": "https://cwe.mitre.org/data/definitions/200.html",
    "CWE-276": "https://cwe.mitre.org/data/definitions/276.html",
    "CWE-312": "https://cwe.mitre.org/data/definitions/312.html",
    "CWE-532": "https://cwe.mitre.org/data/definitions/532.html",
    "CWE-676": "https://cwe.mitre.org/data/definitions/676.html",
    "CWE-749": "https://cwe.mitre.org/data/definitions/749.html",
    "CWE-921": "https://cwe.mitre.org/data/definitions/921.html",
    "CWE-925": "https://cwe.mitre.org/data/definitions/925.html",
    "CWE-939": "https://cwe.mitre.org/data/definitions/939.html",
    "other": "https://cwe.mitre.org/index.html",
    "": "Non-vulnerable code"
}

CWE_Mitigation = {
    "CWE-89": "The software constructs all or part of an SQL command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command when it is sent to a downstream component.",
    "CWE-200": "The product exposes sensitive information to an actor that is not explicitly authorized to have access to that information.",
    "CWE-276": "During installation, installed file permissions are set to allow anyone to modify those files.",
    "CWE-312": "The application stores sensitive information in cleartext within a resource that might be accessible to another control sphere.",
    "CWE-532": "Try to avoid inserting any confidential information in log statements. Minimise using log files in production-level apps.",
    "CWE-676": "The program invokes a potentially dangerous function that could introduce a vulnerability if it is used incorrectly, but the function can also be used safely.",
    "CWE-749": "The software provides an Applications Programming Interface (API) or similar interface for interaction with external actors, but the interface includes a dangerous method or function that is not properly restricted.",
    "CWE-921": "The software stores sensitive information in a file system or device that does not have built-in access control.",
    "CWE-925": "The Android application uses a Broadcast Receiver that receives an Intent but does not properly verify that the Intent came from an authorized source.",
    "CWE-939": "The software uses a handler for a custom URL scheme, but it does not properly restrict which actors can invoke the handler using the scheme.",
    "other": "Some other vulnerability",
    "": "Non-vulnerable code"
}

with open('binary_model.pickle', 'rb') as bin_model_file:
    binary_vectorizer, binary_classifier = pickle.load(bin_model_file)

with open('multiclass_model.pickle', 'rb') as multi_model_file:
    multi_vectorizer, mutli_classifier, encoder = pickle.load(multi_model_file)


def preprocess_comments_and_strings(code_line):
    processed_code_line = code_line

    encryption_hashing_pattern = "AES|aes|SHA-1|sha-1|SHA1|sha1|MD5|md5"
    ip_pattern = "\w*([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\w*"
    string_pattern = "\"[\w|\s|$|&|+|,|:|;|=|?|@|#|_|/|\-|\.|!|`|~|%|\^|\*|\(|\)|\'\\[|\]\{|\}]*\""

    # Checking for encryption related strings
    find_encryption = re.search(encryption_hashing_pattern, processed_code_line)

    # Checking for IP related strings
    find_ip = re.search(ip_pattern, processed_code_line)

    if (find_encryption is None) & (find_ip is None):
        # replacing all strings with dummy string
        processed_code_line = re.sub(string_pattern, "\"user_str\"", processed_code_line)

    # replacing comments with dummy comment
    # comment_pattern = "//.*|/\\*(?s:.*?)\\*/|(\"(?:(?<!\\\\)(?:\\\\\\\\)*\\\\\"|[^\r\n\"])*\")"
    comment_pattern = "//.*|/\\*(?s:.*?)\\*/|/\\*(.)*|(.)*\\*/"
    processed_code_line = re.sub(comment_pattern, "//user_comment", processed_code_line)

    return processed_code_line


def check_vulnerability(test_code):
    CWE_code_word_list = ""
    predicted_cwe_category_probability = 0
    xt_binary = binary_vectorizer.transform([test_code])
    is_vulnearble = binary_classifier.predict(xt_binary)
    pre_processed_test_code = preprocess_comments_and_strings(test_code)
    vulnerability_probability, vulnerable_code_word_list = show_reason_binary(pre_processed_test_code, is_vulnearble[0])
    vulnerable_class = 0
    vul_nonvul = "Vulnerable Code"
    if (round(is_vulnearble[0][0]) == 1):
        xt_multi = multi_vectorizer.transform([test_code])
        prediction = mutli_classifier.predict(xt_multi)
        decoded_prediction = encoder.inverse_transform(np.argmax(prediction, axis=1))
        vulnerable_class = decoded_prediction[0]
        predicted_cwe_category_probability, CWE_code_word_list = show_reason_multiclass(pre_processed_test_code,
                                                                                        vulnerable_class)

    else:
        vulnerable_class = 0
        vul_nonvul = "Non-Vulnerable Code"
    return pre_processed_test_code, vul_nonvul, vulnerable_class, vulnerability_probability[0], vulnerable_code_word_list, predicted_cwe_category_probability, CWE_code_word_list


def show_reason_binary(test_code, is_vulnerable):
    c = make_pipeline(binary_vectorizer, binary_classifier)
    class_names = [0, 1]
    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(test_code, c.predict, num_features=5, top_labels=1)

    vulnerability_probability = c.predict([test_code])[0]

    vulnerable_code_word_list = ""
    if (is_vulnerable == 1):
        vulnerable_code_word_list = str(exp.as_list())

    return vulnerability_probability, vulnerable_code_word_list


def show_reason_multiclass(test_code, vulnerable_class):
    c = make_pipeline(multi_vectorizer, mutli_classifier)
    class_names = ["CWE-200", "CWE-276", "CWE-312", "CWE-532", "Other", "CWE-676", "CWE-749", "CWE-89", "CWE-921",
                   "CWE-985", "CWE-939"]
    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(test_code, c.predict, num_features=5, top_labels=1)

    cwe_class_name_index = class_names.index(vulnerable_class)
    predicted_cwe_category_probability = c.predict([test_code])[0, cwe_class_name_index].round(2)

    CWE_code_word_list = ""

    if predicted_cwe_category_probability > 0.75:
        CWE_code_word_list = exp.as_list(label=cwe_class_name_index)

    return predicted_cwe_category_probability, CWE_code_word_list


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/')
def index():
    return "welcome to vulnerability checker"

@app.route('/code', methods=['GET'])
def get():
    return_string = ""
    for test_code in test_data:
        return_string = return_string + str(check_vulnerability(test_code))
    return return_string

@app.route('/code/<string:test_code>', methods=['GET'])
def getCWE(test_code):
    pre_processed_test_code, vul_nonvul, vulnerable_class, vulnerability_probability, vulnerable_code_word_list, predicted_cwe_category_probability, CWE_code_word_list = check_vulnerability(
        test_code)
    from collections import OrderedDict

    if vulnerable_class == 0:
        vulnerable_class = ""
    return jsonify({'code': str(test_code),
                    'processed_code': str(pre_processed_test_code),
                    'code_vulnerability_status': str(vul_nonvul),
                    'code_vulnerability_probability': str(vulnerability_probability),
                    'probability_breakdown_of_vulnerable_code_words': str(vulnerable_code_word_list),
                    'cwe_id': str(vulnerable_class),
                    'predicted_cwe_id_probability': str(predicted_cwe_category_probability),
                    'probability_breakdown_of_cwe_related_vulnerable_code_words': str(CWE_code_word_list),
                    'description': str(CWE_Desc[str(vulnerable_class)]),
                    'mitigation': str(CWE_Mitigation[str(vulnerable_class)]),
                    'cwe_reference': str(CWE_Links[str(vulnerable_class)])
                    })

    # run_simple('localhost', 5000, app)


def run_api():
    print("FedREVAN API started: http://localhost:5000/")
    serve(app, host="localhost", port=5000)


if __name__ == "__main__":
    run_api()
