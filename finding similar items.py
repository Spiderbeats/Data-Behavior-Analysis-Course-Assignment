import jieba
import re
import random
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from collections import defaultdict
import os
# --- 新增：用于余弦相似度验证 ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Academic Style Configuration ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11


class LSHManager:
    """
    Manually implements the Shingling-MinHash-LSH pipeline for text duplicate detection.
    Core algorithms are implemented without third-party LSH libraries.
    """

    def __init__(self, p=4294967311):
        # p is a large prime number for universal hashing to minimize collisions
        self.p = p
        # Define a standard stopword set to filter low-information tokens
        self.stop_words = {'的', '了', '在', '是', '和', '就', '都', '而', '及', '与', '等', '之', '一'}

    def preprocess(self, text):
        """
        Cleans raw text by removing non-alphanumeric characters and stopwords.
        :param text: Raw input string.
        :return: A list of cleaned tokens.
        """
        if not text or not isinstance(text, str):
            return []
        # Remove punctuation and special characters using regex
        text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
        # Tokenization using jieba
        words = [w for w in jieba.lcut(text) if w.strip()]
        # Stopword removal
        return [w for w in words if w not in self.stop_words]

    def get_shingles(self, words, k, mode='char'):
        """
        Generates k-shingles and maps them to integers via hashing.
        :param words: List of preprocessed tokens.
        :param k: Length of each shingle.
        :param mode: 'char' for character-level or 'word' for word-level shingling.
        :return: A set of integer hashes representing the document.
        """
        source = "".join(words) if mode == 'char' else words
        # Robustness Check: If k exceeds source length, return empty set
        if k > len(source) or k <= 0:
            return set()

        shingles = set()
        for i in range(len(source) - k + 1):
            shingle_str = "".join(source[i:i + k])
            # Manual hash compression using MD5 to map string shingles to integers
            h_int = int(hashlib.md5(shingle_str.encode('utf-8')).hexdigest(), 16) % self.p
            shingles.add(h_int)
        return shingles

    def build_minhash_matrix(self, all_shingles_sets, n_perms):
        """
        Constructs the Min-Hash signature matrix using universal hashing functions.
        Simulates random permutations: h(x) = (a*x + b) % p
        :param all_shingles_sets: List of sets containing shingle hashes for each document.
        :param n_perms: Number of hash functions (permutations N).
        :return: A signature matrix of shape (n_perms, n_docs).
        """
        n_docs = len(all_shingles_sets)
        # Seed the random generator for reproducibility in scientific reporting
        random.seed(42)
        a_coeffs = random.sample(range(1, self.p), n_perms)
        b_coeffs = random.sample(range(1, self.p), n_perms)

        # Initialize signature matrix with infinity
        sig_matrix = np.full((n_perms, n_docs), np.inf)

        for doc_idx, shingle_set in enumerate(all_shingles_sets):
            # Skip empty documents to ensure robustness
            if not shingle_set:
                continue
            for val in shingle_set:
                for i in range(n_perms):
                    # Universal hashing to simulate independent permutations
                    hash_val = (a_coeffs[i] * val + b_coeffs[i]) % self.p
                    if hash_val < sig_matrix[i, doc_idx]:
                        sig_matrix[i, doc_idx] = hash_val
        return sig_matrix

    def get_lsh_candidates(self, sig_matrix, b, r):
        """
        Implements LSH banding to find candidate pairs in sub-linear time.
        :param sig_matrix: The Min-Hash signature matrix.
        :param b: Number of bands.
        :param r: Rows per band. (Ensure b * r <= N)
        :return: A set of candidate pairs (tuples).
        """
        n_perms, n_docs = sig_matrix.shape
        # Adjust r if the product b*r exceeds the matrix rows
        r = min(r, n_perms // b)
        candidates = set()

        for band_idx in range(b):
            # Hash buckets for the current band
            buckets = defaultdict(list)
            start, end = band_idx * r, (band_idx + 1) * r
            for doc_idx in range(n_docs):
                # Core LSH logic: slice the signature into bands and use as bucket key
                band_sig = tuple(sig_matrix[start:end, doc_idx])
                buckets[hash(band_sig)].append(doc_idx)

            # Identify collisions within the same bucket
            for doc_list in buckets.values():
                if len(doc_list) > 1:
                    for i in range(len(doc_list)):
                        for j in range(i + 1, len(doc_list)):
                            pair = tuple(sorted((doc_list[i], doc_list[j])))
                            candidates.add(pair)
        return candidates


def run_experiment(file_list, k_val=2, n_perms=100, b_val=20, r_val=5, s_thresh=0.25):
    """
    Orchestrates the full duplicate detection workflow with visualization.
    All key parameters are exposed here for high configurability.
    """
    # Initialize the manager
    mgr = LSHManager()

    # 1. Load and Preprocess with Error Handling
    contents = []
    for f in file_list:
        if not os.path.exists(f):
            print(f"Warning: File {f} not found. Skipping.")
            contents.append("")
            continue
        with open(f, "r", encoding="utf-8") as file:
            contents.append(file.read())

    # 2. Shingling Stage
    doc_words = [mgr.preprocess(c) for c in contents]

    # --- 修改点 1: 步骤 1 的 k 值对比 ---
    print("\n[Step 1: Impact of k-shingle size on Jaccard Similarity]")
    for k_test in [2, 5, 10]:
        test_shingles = [mgr.get_shingles(w, k_test, 'char') for w in doc_words]
        if len(test_shingles) >= 2:
            s1, s2 = test_shingles[0], test_shingles[1]
            sim = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
            print(f"k = {k_test:<2} | Jaccard(Doc1, Doc2) = {sim:.4f}")
    print("-" * 50)

    all_shingles = [mgr.get_shingles(w, k_val, 'char') for w in doc_words]

    # 3. Compute Ground Truth Jaccard Similarity
    actual_sims = {}
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            set_i, set_j = all_shingles[i], all_shingles[j]
            if not set_i and not set_j:
                actual_sims[(i, j)] = 0.0
            else:
                actual_sims[(i, j)] = len(set_i & set_j) / len(set_i | set_j)

    ground_truth = {p for p, s in actual_sims.items() if s >= s_thresh}
    total_neg = (len(file_list) * (len(file_list) - 1) / 2) - len(ground_truth)

    # --- Visualization Section ---
    print(f"Executing LSH experiment [N={n_perms}, b={b_val}, r={r_val}]...")

    # Plot 1: Estimation Error vs N
    n_tests = [20, 50, 100, 150, 200, 300]
    n_errors = []
    for n in n_tests:
        sig_mat = mgr.build_minhash_matrix(all_shingles, n)
        err = [abs(actual_sims[p] - np.mean(sig_mat[:, p[0]] == sig_mat[:, p[1]])) for p in actual_sims]
        n_errors.append(np.mean(err))

    plt.figure(figsize=(8, 5))
    plt.plot(n_tests, n_errors, marker='s', color='#1f77b4', linewidth=2)
    plt.title("Min-Hash Estimation Error vs. N", fontweight='bold')
    plt.xlabel("Number of Permutations (N)");
    plt.ylabel("Mean Absolute Error");
    plt.grid(True, linestyle='--')
    plt.savefig("plot_n_error.png", dpi=300);
    plt.close()

    # Plot 2: LSH metrics for various (b, r)
    sig_mat_fixed = mgr.build_minhash_matrix(all_shingles, n_perms)
    # Automated testing of different banding configurations
    configs = [(1, n_perms), (4, n_perms // 4), (10, n_perms // 10), (20, n_perms // 20), (25, n_perms // 25)]
    fpr_list, fnr_list, x_labels = [], [], []
    for b, r in configs:
        cands = mgr.get_lsh_candidates(sig_mat_fixed, b, r)
        fp = len([c for c in cands if c not in ground_truth])
        fn = len([gt for gt in ground_truth if gt not in cands])
        fpr_list.append(fp / total_neg if total_neg > 0 else 0)
        fnr_list.append(fn / len(ground_truth) if ground_truth else 0)
        x_labels.append(f"({b},{r})")

    plt.figure(figsize=(10, 5))
    plt.plot(x_labels, fpr_list, 'r-o', label='FPR (False Positives)')
    plt.plot(x_labels, fnr_list, 'g-^', label='FNR (False Negatives)')
    plt.title("LSH Error Rates Across Banding Configurations", fontweight='bold')
    plt.xlabel("(b, r) Config");
    plt.ylabel("Rate");
    plt.legend();
    plt.grid(axis='y')
    plt.savefig("plot_lsh_metrics.png", dpi=300);
    plt.close()

    # Plot 3: Jaccard Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(actual_sims.values(), bins=10, color='#1f77b4', edgecolor='white')
    plt.axvline(s_thresh, color='orange', linestyle='--', label=f'Threshold s={s_thresh}')
    plt.title("Distribution of Pairwise Jaccard Similarities", fontweight='bold')
    plt.xlabel("Jaccard Score");
    plt.ylabel("Frequency");
    plt.legend()
    plt.savefig("plot_jaccard_dist.png", dpi=300);
    plt.close()

    # Plot 4: Theoretical Curve
    plt.figure(figsize=(8, 6))
    t_vals = np.linspace(0, 1, 100)
    for b, r in [(20, 5), (10, 10), (5, 20)]:
        p_vals = 1 - (1 - t_vals ** r) ** b
        plt.plot(t_vals, p_vals, label=f"b={b}, r={r}")
    plt.title("LSH Theoretical Probability (S-Curve)", fontweight='bold')
    plt.xlabel("Actual Similarity (t)");
    plt.ylabel("Probability P");
    plt.legend();
    plt.grid(True)
    plt.savefig("plot_theoretical_scurve.png", dpi=300);
    plt.close()

    # 详细结果打印 (Detailed Terminal Output) ---
    print("\n" + "=" * 50)
    print("      LSH EXPERIMENT DETAILED REPORT")
    print("=" * 50)

    # 1. 参数概览
    print(f"[Parameters] k: {k_val}, N: {n_perms}, b: {b_val}, r: {r_val}")

    # 2. 核心相似度对比 (Top Pairs)
    print("\n[Top Similar Pairs - Ground Truth vs. Min-Hash]")
    print(f"{'Pairs':<15} | {'Actual Jaccard':<15} | {'Min-Hash Sim':<15}")
    print("-" * 50)

    sig_mat_final = mgr.build_minhash_matrix(all_shingles, n_perms)
    for (i, j), actual_s in sorted(actual_sims.items(), key=lambda x: x[1], reverse=True)[:5]:
        # 计算当前的签名相似度
        minhash_s = np.mean(sig_mat_final[:, i] == sig_mat_final[:, j])
        print(f"Doc{i + 1}-Doc{j + 1:<8} | {actual_s:<15.4f} | {minhash_s:<15.4f}")

    # 3. LSH 性能指标
    final_cands = mgr.get_lsh_candidates(sig_mat_final, b_val, r_val)
    tp = len([c for c in final_cands if c in ground_truth])
    fp = len([c for c in final_cands if c not in ground_truth])
    fn = len([gt for gt in ground_truth if gt not in final_cands])

    print("\n[LSH Performance Metrics]")
    print(f"- Total Candidate Pairs Found: {len(final_cands)}")
    print(f"- True Positives (TP): {tp}")
    print(f"- False Positives (FP): {fp}")
    print(f"- False Negatives (FN): {fn}")
    print(f"- Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
    print(f"- Recall (Sensitivity): {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")

    # --- 修改点 2: 步骤 4 的余弦相似度精准验证 ---
    print("\n[Step 4: Cosine Similarity Verification for Candidates]")
    print(f"{'Candidate Pair':<15} | {'Cosine Sim':<15} | {'Is Truly Similar?'}")
    print("-" * 60)

    vectorizer = TfidfVectorizer()
    valid_contents = [c if c.strip() else " " for c in contents]
    tfidf_matrix = vectorizer.fit_transform(valid_contents)

    for (i, j) in sorted(final_cands):
        cos_sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
        is_similar = "YES" if cos_sim > 0.3 else "NO"
        print(f"Doc{i+1}-Doc{j+1:<8} | {cos_sim:<15.4f} | {is_similar}")

    print("=" * 50)
    print("\nExperiment complete. Analysis plots generated.")


if __name__ == "__main__":
    # Define files and parameters here
    target_files = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]
    # All key parameters (k, N, b, r, s) can be adjusted in the line below:
    run_experiment(target_files, k_val=2, n_perms=100, b_val=25, r_val=4, s_thresh=0.25)
