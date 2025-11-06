def number_to_english(num):
    """
    将阿拉伯数字转换为英文读法
    支持整数和小数
    """
    # 转换为字符串以便处理小数
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
        result = 'negative '
    
    result += convert_integer_english(integer_num)
    
    # 转换小数部分
    if decimal_part:
        result += ' point'
        for digit_char in decimal_part:
            result += ' ' + convert_digit_english(int(digit_char))
    
    return result


def convert_digit_english(digit):
    """
    转换单个数字(0-9)为英文
    """
    if isinstance(digit, str):
        digit = int(digit)
    digits = ['zero', 'one', 'two', 'three', 'four', 
              'five', 'six', 'seven', 'eight', 'nine']
    return digits[digit]


def convert_integer_english(num):
    """
    转换整数部分为英文
    """
    if num == 0:
        return 'zero'
    num = int(num)
    # 处理十亿级
    billion = num // 1000000000
    million = (num % 1000000000) // 1000000
    thousand = (num % 1000000) // 1000
    remainder = num % 1000
    
    result = []
    
    # 十亿
    if billion > 0:
        result.append(convert_hundreds(billion) + ' billion')
    
    # 百万
    if million > 0:
        result.append(convert_hundreds(million) + ' million')
    
    # 千
    if thousand > 0:
        result.append(convert_hundreds(thousand) + ' thousand')
    
    # 余数（百位以下）
    if remainder > 0:
        result.append(convert_hundreds(remainder))
    
    return ' '.join(result)


def convert_hundreds(num):
    """
    转换三位数以内的数字（0-999）
    """
    if num == 0:
        return ''
    
    # 1-19的特殊单词
    ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
            'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 
            'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    
    # 20-90的整十单词
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 
            'sixty', 'seventy', 'eighty', 'ninety']
    
    hundred = int(num // 100)
    remainder = int(num % 100)
    
    result = []
    
    # 百位
    if hundred > 0:
        result.append(ones[hundred] + ' hundred')
    
    # 十位和个位
    if remainder >= 20:
        ten = int(remainder // 10)
        one = int(remainder % 10)
        if one > 0:
            result.append(tens[ten] + '-' + ones[one])
        else:
            result.append(tens[ten])
    elif remainder > 0:
        result.append(ones[remainder])
    
    return ' '.join(result)


def convert_with_and(num):
    """
    转换整数为英文（英式英语，使用 "and"）
    例如：101 -> "one hundred and one"
    """
    num_str = str(num)
    
    # 分离整数部分和小数部分
    if '.' in num_str:
        integer_part, decimal_part = num_str.split('.')
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
    
    result = ''
    if is_negative:
        result = 'negative '
    
    # 特殊处理零
    if integer_num == 0:
        result += 'zero'
    else:
        # 处理各级单位
        billion = integer_num // 1000000000
        million = (integer_num % 1000000000) // 1000000
        thousand = (integer_num % 1000000) // 1000
        remainder = integer_num % 1000
        
        parts = []
        
        if billion > 0:
            parts.append(convert_hundreds(billion) + ' billion')
        
        if million > 0:
            parts.append(convert_hundreds(million) + ' million')
        
        if thousand > 0:
            parts.append(convert_hundreds(thousand) + ' thousand')
        
        # 如果有更高位且余数小于100，添加 "and"
        if remainder > 0:
            if len(parts) > 0 and remainder < 100:
                parts.append('and ' + convert_hundreds(remainder))
            else:
                parts.append(convert_hundreds(remainder))
        
        result += ' '.join(parts)
    
    # 小数部分
    if decimal_part:
        result += ' point'
        for digit_char in decimal_part:
            result += ' ' + convert_digit_english(int(digit_char))
    
    return result


# 测试程序
if __name__ == "__main__":
    # 测试用例
    test_numbers = [
        0, 1, 10, 11, 20, 21, 99, 100, 101, 123, 1001, 1010, 1234, 
        10000, 10001, 100000, 123456, 1000000, 12345678, 
        100000000, 1234567890, 123456789012,
        # 小数测试
        3.14, 10.05, 0.5, 123.456, -5.5, 1010.01, 100000.001
    ]
    
    print("=" * 70)
    print("美式英语读法（不使用 and）：")
    print("=" * 70)
    for num in test_numbers:
        print(f"{str(num):>15} -> {number_to_english(num)}")
    
    print("\n" + "=" * 70)
    print("英式英语读法（使用 and）：")
    print("=" * 70)
    # 显示几个典型的差异例子
    examples = [101, 1001, 123, 1234, 100000.001]
    for num in examples:
        print(f"{str(num):>15} -> {convert_with_and(num)}")
    
    print("\n" + "=" * 70)
    print("请输入阿拉伯数字（支持小数，输入q退出）：")
    print("输入 'style' 可切换美式/英式读法")
    print("=" * 70)
    
    use_and = False  # 默认使用美式（不加and）
    
    while True:
        style_hint = "(英式)" if use_and else "(美式)"
        user_input = input(f"{style_hint} > ")
        
        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 'style':
            use_and = not use_and
            style_name = "英式（使用 and）" if use_and else "美式（不使用 and）"
            print(f"已切换到{style_name}\n")
            continue
        
        try:
            num = float(user_input)
            if use_and:
                english = convert_with_and(num)
            else:
                english = number_to_english(num)
            print(f"结果：{english}\n")
        except ValueError:
            print("请输入有效的数字！\n")