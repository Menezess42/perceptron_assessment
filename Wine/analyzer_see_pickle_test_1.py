import pandas as pd
from rich.console import Console
from rich.table import Table

# Carrega o DataFrame
df = pd.read_pickle("./Reports/reports_test_1/compiled_data_test_1.pkl")

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

