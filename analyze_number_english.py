import re
from typing import Tuple, List, Dict
from read_number_english import number_to_english, convert_digit_english
class EnglishNumberReader:
    """英文数字读法判断器"""
    
    def __init__(self):
        # 定义关键词模式
        self.phone_keywords = ['phone', 'call', 'telephone', 'mobile', 'cell', 'contact']
        self.year_keywords = ['year', 'in', 'since', 'until', 'by']
        self.id_keywords = ['id', 'number', 'code', 'reference', 'account', 'card', 'license', 'passport', 'ssn']
        self.quantity_units = ['dollars', 'pounds', 'euros', 'years', 'months', 'days', 'hours', 
                              'meters', 'miles', 'kilometers', 'kg', 'lbs', 'people', 'items']
        self.ordinal_indicators = ['st', 'nd', 'rd', 'th', 'first', 'second', 'third']
    
    def should_read_digit_by_digit(self, text: str, number: str, 
                                   context_before: str, context_after: str) -> Tuple[bool, str]:
        """
        判断数字应该按位读还是按基数读
        
        返回: (是否按位读, 读法类型说明)
        读法类型:
        - "digits": 按位读 (e.g., "one three eight zero zero")
        - "cardinal": 基数读法 (e.g., "thirteen thousand eight hundred")
        - "ordinal": 序数读法 (e.g., "first", "twenty-third")
        - "year": 年份读法 (e.g., "twenty twenty-four" or "two thousand twenty-four")
        - "decimal": 小数读法 (e.g., "three point one four")
        - "money": 金额读法 (e.g., "one hundred ninety-nine dollars")
        """
        
        # 预处理
        number_clean = number.replace(',', '').replace(' ', '')
        context_before_lower = context_before.lower().strip()
        context_after_lower = context_after.lower().strip()
        
        # 1. 小数判断
        if '.' in number:
            return False, "decimal"
        
        # 2. 电话号码判断
        if self._is_phone_number(number_clean, context_before_lower, context_after_lower):
            return True, "digits"
        
        # 3. 年份判断
        if self._is_year(number_clean, context_before_lower, context_after_lower):
            return False, "year"
        
        # 4. 证件号/编号判断
        if self._is_id_number(number_clean, context_before_lower, context_after_lower):
            return True, "digits"
        
        # 5. 序数判断
        if self._is_ordinal(number, context_before_lower, context_after_lower):
            return False, "ordinal"
        
        # 6. 金额判断
        if self._is_money(context_before_lower, context_after_lower):
            return False, "money"
        
        # 7. 房间号/门牌号
        if self._is_room_number(context_before_lower, context_after_lower):
            # 短号码倾向按位读
            if len(number_clean) <= 4:
                return True, "digits"
            return False, "cardinal"
        
        # 8. 时间（小时）
        if self._is_time(context_after_lower):
            return False, "cardinal"
        
        # 9. 数量/度量
        if self._is_quantity(context_after_lower):
            return False, "cardinal"
        
        # 10. 前导零判断
        if number_clean.startswith('0') and len(number_clean) > 1:
            return True, "digits"
        
        # 11. 长数字默认按位读
        if len(number_clean) > 7:
            return True, "digits"
        
        # 12. 默认基数读法
        return False, "cardinal"
    
    def _is_phone_number(self, number: str, before: str, after: str) -> bool:
        """判断是否为电话号码"""        
        # 关键词判断
        if any(kw in before for kw in self.phone_keywords):
            return True
        
        # 格式特征 (xxx-xxxx or xxx xxx xxxx)
        if re.search(r'\d{3}[-\s]?\d{3,4}[-\s]?\d{4}', before + number + after):
            return True
        
        # 长度特征
        if len(number) not in [7, 10, 11]:
            return False
        return False
    
    def _is_year(self, number: str, before: str, after: str) -> bool:
        """判断是否为年份"""
        if len(number) != 4:
            return False
        
        try:
            year = int(number)
            if not (1000 <= year <= 2100):
                return False
            
            # 检查年份相关词汇
            if any(kw in before[-20:] or kw in after[:10] for kw in self.year_keywords):
                return True
            
            # 检查日期格式 (2024/01/01 or January 2024)
            if re.search(r'(19|20)\d{2}', before + number + after):
                return True
            
        except ValueError:
            return False
        
        return False
    
    def _is_id_number(self, number: str, before: str, after: str) -> bool:
        """判断是否为证件号/编号"""
        # 关键词判断
        if any(kw in before for kw in self.id_keywords):
            return True
        
        # 长号码特征（如身份证、护照号等）
        if len(number) >= 8 and any(kw in after for kw in ['number', 'no.', 'no']):
            return True
        
        return False
    
    def _is_ordinal(self, number: str, before: str, after: str) -> bool:
        """判断是否为序数"""
        # 序数后缀
        if re.search(r'\d+(st|nd|rd|th)\b', number + after[:3]):
            return True
        
        # "the + number" 模式
        if 'the' in before[-5:] and any(ind in after[:10] for ind in ['place', 'position', 'rank']):
            return True
        
        return False
    
    def _is_money(self, before: str, after: str) -> bool:
        """判断是否为金额"""
        currency_symbols = ['$', '£', '€', '¥', 'usd', 'gbp', 'eur']
        
        if any(symbol in before[-5:] for symbol in currency_symbols):
            return True
        
        if any(unit in after[:15] for unit in ['dollars', 'pounds', 'euros', 'yuan', 'cents']):
            return True
        
        return False
    
    def _is_room_number(self, before: str, after: str) -> bool:
        """判断是否为房间号"""
        room_keywords = ['room', 'apartment', 'apt', 'suite', 'floor', 'unit']
        return any(kw in before[-10:] or kw in after[:10] for kw in room_keywords)
    
    def _is_time(self, after: str) -> bool:
        """判断是否为时间"""
        time_indicators = ["o'clock", 'am', 'pm', 'a.m.', 'p.m.', 'hours']
        return any(ind in after[:10] for ind in time_indicators)
    
    def _is_quantity(self, after: str) -> bool:
        """判断是否为数量"""
        return any(unit in after[:20] for unit in self.quantity_units)


def analyze_english_text(text: str) -> List[Dict]:
    """
    分析英文文本中的所有数字
    
    返回: 数字分析结果列表
    """
    reader = EnglishNumberReader()
    
    # 匹配数字（包括小数、逗号分隔、序数后缀）
    pattern = r'(?<!\d)(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(st|nd|rd|th)?(?!\d)'
    
    results = []
    for match in re.finditer(pattern, text):
        number = match.group(1)
        ordinal_suffix = match.group(2) or ''
        full_number = number + ordinal_suffix
        
        start = match.start()
        end = match.end()
        
        context_before = text[max(0, start-30):start]
        context_after = text[end:min(len(text), end+30)]
        
        is_digit, read_type = reader.should_read_digit_by_digit(
            text, number, context_before, context_after
        )
        #阿拉伯数字和英文的位数不相等，导致以下操作会错位
        pre_number = number.replace(",","")
        if is_digit:
            tmp = ""
            for c in pre_number:
                tmp += convert_digit_english(int(c)) + " "
            text = text.replace(number, tmp)
        else:
            text = text.replace(number, number_to_english(pre_number))
                        
        results.append({
            'number': full_number,
            'is_digit_by_digit': is_digit,
            'read_type': read_type,
            'context_before': context_before,
            'context_after': context_after,
            'explanation': get_reading_explanation(full_number, read_type)
        })
    
    return text, results


def get_reading_explanation(number: str, read_type: str) -> str:
    """生成读法示例"""
    explanations = {
        'digits': f'Read digit by digit: {number}',
        'cardinal': f'Read as cardinal number: {number}',
        'ordinal': f'Read as ordinal: {number}',
        'year': f'Read as year: {number}',
        'decimal': f'Read with decimal point: {number}',
        'money': f'Read as money amount: {number}'
    }
    return explanations.get(read_type, '')


# 测试用例
def run_tests():
    test_cases = [
        "Please call me at 555-123-4567",
        "I am 25 years old",
        "Room 305 is on the third floor",
        "The year 2024 was eventful",
        "Pi is approximately 3.14159",
        "There are 1,234 apples in the basket",
        "My passport number is A12345678",
        "He finished in 3rd place with a time of 10.5 seconds",
        "The total is $199.99",
        "Flight number BA0123 departs at 3:45 PM",
        "The population is 7,500,000 people",
        "Born in 1985, graduated in 2007",
        "Temperature is 98.6 degrees",
        "The 21st century began in 2001",
        "Product code: SKU-0012345",
        "He lives at 10 Downing Street",
        "The price increased by 15 percent",
        "Call 911 for emergency",
    ]
    
    print("=" * 80)
    print("ENGLISH NUMBER READING ANALYSIS")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {case}")
        text, results = analyze_english_text(case)
        
        for r in results:
            read_method = "DIGIT-BY-DIGIT" if r['is_digit_by_digit'] else "AS NUMBER"
            print(f"  → Number: {r['number']}")
            print(f"    Method: {read_method}")
            print(f"    Type: {r['read_type']}")
            print(f"    Example: {r['explanation']}")
        print("text", text)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_tests()