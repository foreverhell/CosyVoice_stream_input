def number_to_chinese(num):
    """
    将阿拉伯数字转换为中文读法
    支持整数和小数
    """
    # 数字对应的中文
    digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    
    num_str = ""
    # 转换为字符串以便处理小数
    if isinstance(num, str):
        num_str = num
        num = float(num)
    elif isinstance(num, float):
        num_str = str(num)
    
    # 分离整数部分和小数部分
    if '.' in num_str:
        integer_part, decimal_part = num_str.split('.')
        # 处理负数
        if integer_part.startswith('-'):
            integer_part = integer_part[1:]
            is_negative = True
        else:
            is_negative = False
        integer_num = int(integer_part) if integer_part else 0
    else:
        if num < 0:
            is_negative = True
            integer_num = -num
        else:
            is_negative = False
            integer_num = num
        decimal_part = None
    
    # 转换整数部分
    result = ''
    if is_negative:
        result = '负'
    
    result += convert_integer(integer_num, digits)
    
    # 转换小数部分
    if decimal_part:
        result += '点'
        for digit_char in decimal_part:
            result += digits[int(digit_char)]
    
    return result

def convert_integer(num, digits):
    """
    将阿拉伯数字转换为中文读法
    支持范围：0 到 9999亿（兆级以内）
    """
    # 数字对应的中文
    digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    
    num = int(num)
    
    if num == 0:
        return '零'
    
    if num < 0:
        return '负' + convert_integer(-num)
    
    # 处理亿级
    yi = num // 100000000
    wan = (num % 100000000) // 10000
    ge = num % 10000
    
    result = ''
    
    # 处理亿
    if yi > 0:
        yi_str = convert_section(yi, digits)
        if yi < 20 and yi >= 10:
            yi_str = yi_str[1:]  # 去掉开头的"一"
        result += yi_str + '亿'
        # 如果万级为0但个级不为0，需要加零
        if wan == 0 and ge > 0:
            result += '零'
    
    # 处理万
    if wan > 0:
        # 如果亿级存在且万级不足四位数（小于1000），需要加零
        if yi > 0 and wan < 1000:
            result += '零'        
        wan_str = convert_section(wan, digits)
        if wan < 20 and wan >= 10:
            wan_str = wan_str[1:]  # 去掉开头的"一"
        result += wan_str + '万'
        # 如果个级不足四位数（小于1000）且个级不为0，需要加零
        if ge > 0 and ge < 1000:
            result += '零'
    
    # 处理个级
    if ge > 0:
        # 如果有更高位，且个级小于1000，前面已经加过零了
        ge_str = convert_section(ge, digits)
        # 特殊处理：如果整个数字只有个级，且是10-19，要简化"一十"为"十"
        if num < 20 and num >= 10:
            ge_str = ge_str[1:]  # 去掉开头的"一"
        result += ge_str
    
    return result


def convert_section(num, digits):
    """
    转换四位数以内的数字（0-9999）
    num: 要转换的数字
    digits: 数字对应的中文列表
    """
    if num == 0:
        return ''
    
    qian = num // 1000
    bai = (num % 1000) // 100
    shi = (num % 100) // 10
    ge = num % 10
    
    result = ''
    
    # 千位
    if qian > 0:
        result += digits[qian] + '千'
    
    # 百位
    if bai > 0:
        # 如果千位有值但百位为0，在前面的条件中会处理
        result += digits[bai] + '百'
    elif qian > 0 and (shi > 0 or ge > 0):
        # 千位有值，百位是0，但后面还有数字，需要补零
        result += '零'
    
    # 十位
    if shi > 0:
        result += digits[shi] + '十'
    elif bai > 0 and ge > 0:
        # 百位有值，十位是0，但个位有值，需要补零
        result += '零'
    
    # 个位
    if ge > 0:
        result += digits[ge]
    
    return result


# 测试程序
if __name__ == "__main__":
    # 测试用例
    test_numbers = [0, 1, 10, 11, 20, 123, 1001, 1010, 1234, 10000, 10001, 
                    100000, 123456, 1000000, 12345678, 100000000, 123456789012,
                    # 小数测试
                    3.14, 10.05, 0.5, 123.456, -5.5, 1010.01, 100000.001
                    ]
    
    print("测试结果：")
    for num in test_numbers:
        print(f"{num:>12} -> {number_to_chinese(num)}")
    
    print("\n" + "="*50)
    print("请输入阿拉伯数字（输入q退出）：")
    
    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            break
        try:
            num = float(user_input)
            chinese = number_to_chinese(num)
            print(f"结果：{chinese}\n")
        except ValueError:
            print("请输入有效的整数！\n")