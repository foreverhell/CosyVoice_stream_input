import re
from read_number import number_to_chinese
def read_by_digit(text):
    if isinstance(text, int):
        text = str(text)
        
    digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    result = ""
    for digit_char in text:
        result += digits[int(digit_char)]
    return result

def should_read_digit_by_digit(text, number, context_before, context_after):
    """判断数字是否应该按位读"""
    
    num_str = str(number).replace('.', '')
    has_leading_zero = num_str.startswith('0') and len(num_str) > 1
    digit_count = len(num_str)
    
    # 小数判断。number_to_chinese有处理小数的逻辑，所以按十进制读
    if '.' in str(number):
        return False
    
    # 电话号码，按位读
    if any(kw in context_before for kw in ['电话', '手机', '联系']) or digit_count in [3, 7, 8, 11, 12]:#110, 120这种特殊电话长度为3，座机为7、8（带区号为11、12），手机为11
        return True
    
    # 年份，按位读
    if digit_count == 4:
        try:
            year = int(number)
            if 1900 <= year <= 2100 and '年' in context_after[:3]:
                return True
        except:
            pass
        
    if digit_count == 1 or digit_count == 2:
        try:
            year = int(number)
            if 1 <= year <= 12 and '月' in context_after[:1]:#月，十进制读
                return False
            if 1 <= year <= 31 and ('日' in context_after[:1] or '号' in context_after[:1]):#日，十进制读
                return False
        except:
            pass
        
    # 房间号、编号，按位读
    if any(keyword in context_after[:5] for keyword in ['号', '室', '楼', '单元']):
        if digit_count <= 4:
            return True
    
    # 证件号码，按位读
    if any(keyword in context_before for keyword in ['身份证', '证件', '卡号', '账号', '订单']):
        return True
    
    # 前导零
    if has_leading_zero:
        return True
    
    # 数量单位，十进制读
    quantity_units = ['个', '只', '件', '元', '块', '岁', '人', '天', '米', '公斤', '克']
    if any(unit in context_after[:5] for unit in quantity_units):
        return False
    
    # 序数，十进制读
    if '第' in context_before[-3:]:
        return False
    
    # 金额符号，十进制读
    if any(symbol in context_before[-5:] for symbol in ['¥', '$', '￥', '人民币']):
        return False
    
    # 默认：长数字按位读
    if digit_count > 6:
        return True
    
    return False


def analyze_text(text):
    """分析文本中的所有数字"""
    # 匹配数字（包括小数）
    pattern = r'(?<!\d)(\d+\.?\d*)(?!\d)'
    
    results = []
    for match in re.finditer(pattern, text):
        number = match.group(1)
        start = match.start()
        end = match.end()
        
        context_before = text[max(0, start-15):start]
        context_after = text[end:min(len(text), end+15)]
        is_digit_reading = should_read_digit_by_digit(
            text, number, context_before, context_after
        )
        if is_digit_reading:
            #text = context_before + read_by_digit(number) + context_after
            text = text.replace(number, read_by_digit(number))
        else:
            #text = context_before + number_to_chinese(number) + context_after
            text = text.replace(number, number_to_chinese(number))
        results.append({
            'number': number,
            'type': "按位读" if is_digit_reading else "十进制读",
            'context_before': context_before,
            'context_after': context_after
        })
    
    return text,results

if __name__ == "__main__":
    # 测试
    test_cases = [
        "请拨打电话13800138000",
        "我今年25岁了，它今年12岁",
        "房间号是305室",
        "2024年1月1日",
        "圆周率是3.1415926",
        "一共有1234个苹果",
        "身份证号110101199001011234",
        "第3名获得奖金500元",
        "公交203路",
        "价格是¥199",
        "11月28号晚上9点",
        "帮我电话给110和120"
    ]

    print("=" * 60)
    for case in test_cases:
        print(f"\n原文: {case}")
        text,results = analyze_text(case)
        for r in results:
            print(f"  数字: {r['number']} → {r['type']}")
        print("text",text)
    print("=" * 60)
