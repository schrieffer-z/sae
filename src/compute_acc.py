import json
import argparse
import sys

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='计算JSON文件中有效和正确值的比例')
    parser.add_argument('--file_path', type=str, help='要分析的JSON文件路径')
    parser.add_argument('--latent_path', type=str, help='只计算特定latents的准确性')
    args = parser.parse_args()

    try:
        # 打开并读取JSON文件
        with open(args.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(args.latent_path, 'r', encoding='utf-8') as f:
            selected_latents = json.load(f) 
        
        latent_map = data.get('results', {})
        cnt_valid = 0
        cnt_scored = 0
        cnt_correct = 0
        
        for latent in latent_map:
            # 跳过无效分数条目
            if latent not in selected_latents.keys():
                continue
            if latent_map[latent]['score'] is None:
                continue
            cnt_scored += 1
            if latent_map[latent]['score'] == 0:
                continue

            
            # 统计有效和正确的值
            cnt_valid += 1
            if latent_map[latent]['score'] * selected_latents[latent] > 0:
                cnt_correct += 1
        
        # 输出结果
        if cnt_valid > 0:
            print(f"已评分条目数: {cnt_scored}")
            print(f"有效条目数: {cnt_valid}")
            print(f"正确条目数: {cnt_correct}")
            print(f"正确率: {cnt_correct / cnt_valid:.4f} ({cnt_correct}/{cnt_valid})")
            return 0
        else:
            print("错误: 没有找到有效条目")
            return 1
    
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {args.file_path}")
        return 1
    except json.JSONDecodeError:
        print(f"错误: JSON格式无效 - {args.file_path}")
        return 1
    except KeyError as e:
        print(f"错误: JSON结构缺失键 - {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())