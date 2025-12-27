from pywebio import start_server
from pywebio.input import input, TEXT
from pywebio.output import put_text, put_markdown, put_loading
from agent.agent import query_agent


def main():
    put_markdown("# data-copilot-v5")
    put_text("Please enter your query question")

    while True:
        question = input("Query question:", type=TEXT, required=True)

        if question.strip():
            with put_loading(shape="grow", color="primary"):
                try:
                    answer = query_agent(question,with_exp=True)
                    put_markdown(f"**Query Result:**")
                    put_markdown(answer)
                except Exception as e:
                    put_text(f"Query error: {str(e)}")

            put_markdown("---")
            put_text("You can continue to enter new queries")


if __name__ == '__main__':
    start_server(main, port=8080, debug=True)
    # 查询是否患有 diabetic_retinopathy比例 随年龄的分布

