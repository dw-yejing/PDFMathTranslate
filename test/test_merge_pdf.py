import fitz  # PyMuPDF

def merge_pages_side_by_side(input_path, gap=20):
    """
    将PDF指定页面左右合并
    
    参数:
        input_path: 输入PDF文件路径
        output_path: 输出PDF文件路径
        page_numbers: 要合并的页码列表(从1开始), 例如[1,2]表示合并第1页和第2页
    """
    # 打开原始PDF
    doc = fitz.open(input_path)
    
    output_path = f"{input_path}.merged.pdf"
    total_pages = len(doc)
    
    # 创建一个新的PDF文档
    new_doc = fitz.open()
    
    # 遍历页面，每次处理两页(奇数和偶数页)
    for i in range(0, total_pages, 2):
        # 获取当前页和下一页
        page1 = doc.load_page(i)
        page2 = doc.load_page(i + 1) if i + 1 < total_pages else None
        
        # 计算新页面的尺寸(两页宽度相加，取最大高度)
        width1 = page1.rect.width
        height1 = page1.rect.height
        
        if page2:
            width2 = page2.rect.width
            height2 = page2.rect.height
            new_width = width1 + width2 + gap
            new_height = max(height1, height2)
        else:
            # 如果总页数是奇数，最后一页单独处理
            new_width = width1
            new_height = height1
        
        # 创建新页面
        new_page = new_doc.new_page(width=new_width, height=new_height)
        
        # 将第一页绘制到新页面的左侧
        new_page.show_pdf_page(
            fitz.Rect(0, 0, width1, new_height),  # 目标区域
            doc,  # 源文档
            i  # 页码
        )
        
        # 如果有第二页，绘制到右侧
        if page2:
            new_page.show_pdf_page(
                fitz.Rect(width1+gap, 0, new_width, new_height),  # 目标区域
                doc,  # 源文档
                i + 1  # 页码
            )
    
    # 保存新PDF
    new_doc.save(output_path)
    print(f"处理完成，已保存到: {output_path}")

if __name__ == "__main__":
    # 示例用法
    input_pdf = "a-dual.pdf"  # 替换为你的输入PDF文件
    
    merge_pages_side_by_side(input_pdf)