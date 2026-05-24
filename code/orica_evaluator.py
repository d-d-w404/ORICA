import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.signal import welch
import seaborn as sns

class ORICAEvaluator:
    """ORICA分离效果评估工具"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def comprehensive_evaluation(self, sources, original_data, srate, ica_model):
        """综合评估ORICA分离效果"""
        
        # 1. 统计指标评估
        stats_eval = self._evaluate_statistical_metrics(sources)
        
        # 2. 频谱分析评估
        spectral_eval = self._evaluate_spectral_quality(sources, srate)
        
        # 3. 伪影识别评估
        artifact_eval = self._evaluate_artifact_separation(sources, srate)
        
        # 4. 重构质量评估
        reconstruction_eval = self._evaluate_reconstruction_quality(sources, original_data, ica_model)
        
        # 5. 综合评分
        overall_score = self._calculate_overall_score(stats_eval, spectral_eval, artifact_eval, reconstruction_eval)
        
        # 6. 保存评估历史
        evaluation_result = {
            'stats': stats_eval,
            'spectral': spectral_eval,
            'artifact': artifact_eval,
            'reconstruction': reconstruction_eval,
            'overall_score': overall_score,
            'timestamp': np.datetime64('now')
        }
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def _evaluate_statistical_metrics(self, sources):
        """评估统计指标"""
        # 峰度评估
        kurtosis_values = kurtosis(sources, axis=1, fisher=False)
        avg_kurtosis = np.mean(np.abs(kurtosis_values))
        max_kurtosis = np.max(np.abs(kurtosis_values))
        
        # 互信息评估（使用相关性近似）
        n_components = sources.shape[0]
        correlations = []
        for i in range(n_components):
            for j in range(i+1, n_components):
                corr = np.corrcoef(sources[i], sources[j])[0, 1]
                correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        max_correlation = np.max(correlations) if correlations else 0
        
        return {
            'avg_kurtosis': avg_kurtosis,
            'max_kurtosis': max_kurtosis,
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'kurtosis_values': kurtosis_values
        }
    
    def _evaluate_spectral_quality(self, sources, srate):
        """评估频谱质量"""
        spectral_ratios = []
        
        for i, source in enumerate(sources):
            # 计算功率谱密度
            freqs, psd = welch(source, fs=srate)
            
            # 计算不同频段的功率比例
            delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
            theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
            alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
            beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
            gamma_power = np.sum(psd[(freqs >= 30) & (freqs <= 100)])
            
            total_power = np.sum(psd)
            
            spectral_ratios.append({
                'component': i,
                'delta_ratio': delta_power / total_power,
                'theta_ratio': theta_power / total_power,
                'alpha_ratio': alpha_power / total_power,
                'beta_ratio': beta_power / total_power,
                'gamma_ratio': gamma_power / total_power,
                'total_power': total_power
            })
        
        return {
            'spectral_ratios': spectral_ratios,
            'avg_alpha_ratio': np.mean([r['alpha_ratio'] for r in spectral_ratios]),
            'avg_beta_ratio': np.mean([r['beta_ratio'] for r in spectral_ratios])
        }
    
    def _evaluate_artifact_separation(self, sources, srate):
        """评估伪影分离效果"""
        artifact_scores = []
        
        for i, source in enumerate(sources):
            # 计算眨眼伪影特征
            fft_vals = np.abs(np.fft.rfft(source))
            freqs = np.fft.rfftfreq(source.shape[0], 1/srate)
            
            # 低频功率比例（眨眼特征）
            low_freq_power = np.sum(fft_vals[(freqs >= 0.1) & (freqs <= 4)])
            total_power = np.sum(fft_vals)
            low_freq_ratio = low_freq_power / total_power
            
            # 高频功率比例（肌电伪影特征）
            high_freq_power = np.sum(fft_vals[freqs >= 50])
            high_freq_ratio = high_freq_power / total_power
            
            # 判断是否为伪影
            is_artifact = low_freq_ratio > 0.15 or high_freq_ratio > 0.1
            
            artifact_scores.append({
                'component': i,
                'low_freq_ratio': low_freq_ratio,
                'high_freq_ratio': high_freq_ratio,
                'is_artifact': is_artifact,
                'artifact_type': 'blink' if low_freq_ratio > 0.15 else ('emg' if high_freq_ratio > 0.1 else 'brain')
            })
        
        artifact_count = sum(1 for score in artifact_scores if score['is_artifact'])
        brain_count = len(artifact_scores) - artifact_count
        
        return {
            'artifact_scores': artifact_scores,
            'artifact_count': artifact_count,
            'brain_count': brain_count,
            'artifact_ratio': artifact_count / len(artifact_scores)
        }
    
    def _evaluate_reconstruction_quality(self, sources, original_data, ica_model):
        """评估重构质量"""
        try:
            W = ica_model.get_W()
            A = np.linalg.pinv(W)
            reconstructed = A @ sources
            reconstruction_error = np.mean((original_data - reconstructed)**2)
            
            # 计算信噪比
            signal_power = np.mean(original_data**2)
            noise_power = np.mean((original_data - reconstructed)**2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
        except Exception as e:
            reconstruction_error = float('inf')
            snr = float('-inf')
        
        return {
            'reconstruction_error': reconstruction_error,
            'snr_db': snr
        }
    
    def _calculate_overall_score(self, stats_eval, spectral_eval, artifact_eval, reconstruction_eval):
        """计算综合评分"""
        # 峰度得分 (0-1)
        kurtosis_score = min(stats_eval['avg_kurtosis'] / 5.0, 1.0)
        
        # 独立性得分 (0-1)
        independence_score = max(0, 1 - stats_eval['avg_correlation'] / 0.1)
        
        # 伪影分离得分 (0-1)
        artifact_score = 1 - artifact_eval['artifact_ratio']
        
        # 重构质量得分 (0-1)
        reconstruction_score = max(0, 1 - reconstruction_eval['reconstruction_error'] / 0.01)
        
        # 综合评分
        overall_score = (kurtosis_score * 0.3 + 
                        independence_score * 0.3 + 
                        artifact_score * 0.2 + 
                        reconstruction_score * 0.2)
        
        return overall_score
    
    def print_evaluation_report(self, evaluation_result):
        """打印评估报告"""
        print("\n" + "="*50)
        print("ORICA分离效果评估报告")
        print("="*50)
        
        # 统计指标
        stats = evaluation_result['stats']
        print(f"\n📊 统计指标:")
        print(f"  平均峰度: {stats['avg_kurtosis']:.3f} (目标: >3.0)")
        print(f"  最大峰度: {stats['max_kurtosis']:.3f}")
        print(f"  平均相关性: {stats['avg_correlation']:.3f} (目标: <0.1)")
        print(f"  最大相关性: {stats['max_correlation']:.3f}")
        
        # 频谱质量
        spectral = evaluation_result['spectral']
        print(f"\n📈 频谱质量:")
        print(f"  平均α波比例: {spectral['avg_alpha_ratio']:.3f}")
        print(f"  平均β波比例: {spectral['avg_beta_ratio']:.3f}")
        
        # 伪影分离
        artifact = evaluation_result['artifact']
        print(f"\n👁️ 伪影分离:")
        print(f"  伪影成分数: {artifact['artifact_count']}")
        print(f"  脑电成分数: {artifact['brain_count']}")
        print(f"  伪影比例: {artifact['artifact_ratio']:.2%}")
        
        # 重构质量
        recon = evaluation_result['reconstruction']
        print(f"\n🔄 重构质量:")
        print(f"  重构误差: {recon['reconstruction_error']:.6f} (目标: <0.01)")
        print(f"  信噪比: {recon['snr_db']:.2f} dB")
        
        # 综合评分
        overall = evaluation_result['overall_score']
        print(f"\n🏆 综合评分: {overall:.3f}")
        
        if overall > 0.8:
            print("✅ 分离质量: 优秀")
        elif overall > 0.6:
            print("✅ 分离质量: 良好")
        elif overall > 0.4:
            print("⚠️ 分离质量: 一般")
        else:
            print("❌ 分离质量: 需要改进")
        
        print("="*50)
    
    def plot_evaluation_summary(self, evaluation_result, save_path=None):
        """绘制评估总结图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 峰度分布
        stats = evaluation_result['stats']
        axes[0, 0].bar(range(len(stats['kurtosis_values'])), np.abs(stats['kurtosis_values'])
        axes[0, 0].set_title('Component Kurtosis Distribution')
        axes[0, 0].set_xlabel('Component')
        axes[0, 0].set_ylabel('|Kurtosis|')
        axes[0, 0].axhline(y=3, color='r', linestyle='--', label='Target (3.0)')
        axes[0, 0].legend()
        
        # 2. 频谱比例
        spectral = evaluation_result['spectral']
        ratios = spectral['spectral_ratios']
        alpha_ratios = [r['alpha_ratio'] for r in ratios]
        beta_ratios = [r['beta_ratio'] for r in ratios]
        
        x = range(len(ratios))
        width = 0.35
        axes[0, 1].bar([i - width/2 for i in x], alpha_ratios, width, label='Alpha')
        axes[0, 1].bar([i + width/2 for i in x], beta_ratios, width, label='Beta')
        axes[0, 1].set_title('Spectral Power Ratios')
        axes[0, 1].set_xlabel('Component')
        axes[0, 1].set_ylabel('Power Ratio')
        axes[0, 1].legend()
        
        # 3. 伪影分布
        artifact = evaluation_result['artifact']
        artifact_types = [score['artifact_type'] for score in artifact['artifact_scores']]
        type_counts = {'brain': 0, 'blink': 0, 'emg': 0}
        for t in artifact_types:
            type_counts[t] += 1
        
        axes[1, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Component Type Distribution')
        
        # 4. 评分雷达图
        categories = ['Kurtosis', 'Independence', 'Artifact Separation', 'Reconstruction']
        values = [
            min(stats['avg_kurtosis'] / 5.0, 1.0),
            max(0, 1 - stats['avg_correlation'] / 0.1),
            1 - artifact['artifact_ratio'],
            max(0, 1 - recon['reconstruction_error'] / 0.01)
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        axes[1, 1].plot(angles, values, 'o-', linewidth=2)
        axes[1, 1].fill(angles, values, alpha=0.25)
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Quality Metrics Radar')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"评估图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close() 