import pandas as pd
from rich.console import Console
from rich.table import Table

# Carrega o DataFrame
df = pd.read_pickle("./Reports/reports_test_2/compiled_data_test_2.pkl")
console = Console()


def print_table(data, max_rows=20):
    """Imprime uma tabela Rich formatada."""
    if data.empty:
        console.print("[yellow]Nenhum dado encontrado com os critérios fornecidos.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    for column in data.columns:
        table.add_column(column)

    rows_to_show = data if max_rows is None else data.head(max_rows)
    for _, row in rows_to_show.iterrows():
        table.add_row(*[str(row[col]) for col in data.columns])

    console.print(table)

    if max_rows is not None and len(data) > max_rows:
        console.print(f"[dim]Mostrando apenas as primeiras {max_rows} de {len(data)} linhas.[/dim]")


def show_help():
    console.print("""
[bold cyan]Comandos disponíveis:[/bold cyan]
- [green]columns[/green]: mostra as colunas disponíveis
- [green]filter[/green]: permite visualizar partes específicas dos dados
- [green]help[/green]: mostra esta ajuda
- [green]exit[/green]: sai do programa

[bold cyan]Filtro:[/bold cyan]
- Você pode escolher colunas com: coluna1,coluna2 ou intervalo como: coluna1:coluna4
- E pode escolher faixas de linhas com slice Python: 0:10, -5:, ::2 etc.
""")


def main():
    console.print("[bold green]Sistema de Visualização de Dados Iniciado[/bold green]")
    show_help()

    while True:
        command = console.input("[bold blue]>> [/bold blue]").strip().lower()

        if command == "exit":
            break

        elif command == "help":
            show_help()

        elif command == "columns":
            console.print("[cyan]Colunas disponíveis:[/cyan]", ", ".join(df.columns))

        elif command == "filter":
            col_input = console.input("Colunas (ex: x,y,z ou x:z): ").strip()
            row_input = console.input("Faixa de linhas (ex: 0:10 ou Enter para tudo): ").strip()

            # Processa colunas
            try:
                if ":" in col_input:
                    col_start, col_end = col_input.split(":")
                    col_list = list(df.columns)
                    col_start_idx = col_list.index(col_start)
                    col_end_idx = col_list.index(col_end) + 1
                    columns = col_list[col_start_idx:col_end_idx]
                else:
                    columns = [col.strip() for col in col_input.split(",")]
            except Exception as e:
                console.print(f"[red]Erro ao interpretar colunas: {e}[/red]")
                continue

            # Processa linhas com slice estilo Python
            try:
                if row_input:
                    slice_parts = row_input.split(":")
                    start = int(slice_parts[0]) if slice_parts[0] else None
                    end = int(slice_parts[1]) if len(slice_parts) > 1 and slice_parts[1] else None
                    step = int(slice_parts[2]) if len(slice_parts) > 2 and slice_parts[2] else None
                    slicer = slice(start, end, step)
                    filtered = df.iloc[slicer][columns]
                    max_rows = None  # Mostra tudo que foi pedido
                else:
                    filtered = df.loc[:, columns]
                    max_rows = 20  # Limita se nenhuma linha for especificada
            except Exception as e:
                console.print(f"[red]Erro ao filtrar linhas: {e}[/red]")
                continue

            print_table(filtered, max_rows=max_rows)

        else:
            console.print("[red]Comando inválido. Digite 'help' para ver os comandos disponíveis.[/red]")


if __name__ == "__main__":
    main()

# import pandas as pd
# from rich.console import Console
# from rich.table import Table
# from prompt_toolkit import prompt
# from prompt_toolkit.completion import WordCompleter
#
# # Carrega o DataFrame
# df = pd.read_pickle("./Reports/reports_test_1/compiled_data.pkl")
#
# # Inicializa o console do Rich
# console = Console()
#
# # Comandos disponíveis
# commands = ['show', 'search', 'row', 'columns', 'filter', 'help', 'exit']
# completer = WordCompleter(commands, ignore_case=True)
#
# def print_table(data, max_rows=10):
#     if data.empty:
#         console.print("[bold red]Nenhum resultado encontrado.[/bold red]")
#         return
#
#     table = Table(show_header=True, header_style="bold magenta")
#     for col in data.columns:
#         table.add_column(str(col))
#
#     for _, row in data.head(max_rows).iterrows():
#         table.add_row(*[str(x) for x in row.values])
#
#     console.print(table)
#     if len(data) > max_rows:
#         console.print(f"[dim]Mostrando apenas as primeiras {max_rows} de {len(data)} linhas.[/dim]")
#
# def print_help():
#     console.print("""
# [bold cyan]Comandos disponíveis:[/bold cyan]
#
# [bold yellow]show[/bold yellow]       - Mostra o DataFrame (você escolhe quantas linhas)
# [bold yellow]columns[/bold yellow]    - Lista todas as colunas disponíveis
# [bold yellow]row[/bold yellow]        - Visualiza uma linha pelo índice
# [bold yellow]search[/bold yellow]     - Filtra uma coluna por valor textual
# [bold yellow]filter[/bold yellow]     - Filtra colunas e/ou linhas (faixa ou lista):
#                Ex: colunas x,y,z      -> apenas essas colunas
#                    colunas x:z        -> da coluna x até z
#                    linhas 0:10        -> da linha 0 até 10
#
# [bold yellow]help[/bold yellow]       - Mostra esta ajuda
# [bold yellow]exit[/bold yellow]       - Sai do programa
# """)
#
# def main():
#     console.print("[bold green]Sistema de visualização de dados iniciado[/bold green]")
#     print_help()
#
#     while True:
#         user_input = prompt(">> ", completer=completer).strip().lower()
#
#         if user_input == 'exit':
#             console.print("Encerrando o sistema. Até mais!")
#             break
#
#         elif user_input == 'help':
#             print_help()
#
#         elif user_input == 'show':
#             try:
#                 n = int(prompt("Quantas linhas você quer ver? (default 10): ") or 10)
#                 print_table(df, max_rows=n)
#             except ValueError:
#                 console.print("[red]Número inválido.[/red]")
#
#         elif user_input == 'columns':
#             console.print(f"[bold blue]Colunas disponíveis:[/bold blue] {', '.join(df.columns)}")
#
#         elif user_input == 'row':
#             try:
#                 idx = int(prompt("Digite o índice da linha: "))
#                 if 0 <= idx < len(df):
#                     row = df.iloc[idx]
#                     for key, value in row.items():
#                         console.print(f"[bold]{key}:[/bold] {value}")
#                 else:
#                     console.print("[red]Índice fora do intervalo.[/red]")
#             except ValueError:
#                 console.print("[red]Entrada inválida.[/red]")
#
#         elif user_input == 'search':
#             col = prompt("Coluna para buscar: ").strip()
#             if col not in df.columns:
#                 console.print("[red]Coluna não encontrada.[/red]")
#                 continue
#             val = prompt("Valor a buscar (como string): ").strip()
#             result = df[df[col].astype(str).str.contains(val, case=False, na=False)]
#             print_table(result, max_rows=15)
#
#         elif user_input == 'filter':
#             col_input = prompt("Colunas (ex: x,y,z ou x:z): ").strip()
#             row_input = prompt("Faixa de linhas (ex: 0:10 ou Enter para tudo): ").strip()
#
#             # Processa colunas
#             columns = []
#             if ':' in col_input:
#                 col_list = list(df.columns)
#                 start, end = col_input.split(':')
#                 if start not in col_list or end not in col_list:
#                     console.print("[red]Faixa de colunas inválida.[/red]")
#                     continue
#                 idx_start = col_list.index(start)
#                 idx_end = col_list.index(end) + 1
#                 columns = col_list[idx_start:idx_end]
#             else:
#                 columns = [col.strip() for col in col_input.split(',') if col.strip() in df.columns]
#
#             if not columns:
#                 console.print("[red]Nenhuma coluna válida informada.[/red]")
#                 continue
#
#             # Processa linhas com suporte a slice estilo Python
#             try:
#                 if row_input:
#                     # Usa slice com iloc interpretando os limites
#                     slice_parts = row_input.split(":")
#                     start = int(slice_parts[0]) if slice_parts[0] else None
#                     end = int(slice_parts[1]) if len(slice_parts) > 1 and slice_parts[1] else None
#                     step = int(slice_parts[2]) if len(slice_parts) > 2 and slice_parts[2] else None
#                     slicer = slice(start, end, step)
#                     filtered = df.iloc[slicer][columns]
#                 else:
#                     filtered = df.loc[:, columns]
#             except Exception as e:
#                 console.print(f"[red]Erro ao filtrar linhas: {e}[/red]")
#                 continue
#
#             print_table(filtered, max_rows=20)
#
#         else:
#             console.print("[red]Comando não reconhecido. Digite 'help' para ver opções.[/red]")
#
# if __name__ == "__main__":
#     main()
