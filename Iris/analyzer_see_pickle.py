import pandas as pd
from rich.console import Console
from rich.table import Table
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# Carrega o DataFrame
df = pd.read_pickle("./Reports/reports_test_1/compiled_data.pkl")

# Inicializa o console do Rich
console = Console()

# Completer para facilitar comandos
commands = ['show', 'search', 'row', 'columns', 'exit']
completer = WordCompleter(commands, ignore_case=True)

def print_table(data, max_rows=10):
    if data.empty:
        console.print("[bold red]Nenhum resultado encontrado.[/bold red]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    for col in data.columns:
        table.add_column(str(col))

    for _, row in data.head(max_rows).iterrows():
        table.add_row(*[str(x) for x in row.values])

    console.print(table)
    if len(data) > max_rows:
        console.print(f"[dim]Mostrando apenas as primeiras {max_rows} de {len(data)} linhas.[/dim]")

def main():
    console.print("[bold green]Sistema de visualização de dados iniciado[/bold green]\n")
    while True:
        user_input = prompt("Digite um comando (show, search, row, columns, exit): ", completer=completer).strip().lower()

        if user_input == 'exit':
            console.print("Encerrando o sistema. Até mais!")
            break

        elif user_input == 'show':
            try:
                n = int(prompt("Quantas linhas você quer ver? (default 10): ") or 10)
                print_table(df, max_rows=n)
            except ValueError:
                console.print("[red]Número inválido.[/red]")

        elif user_input == 'columns':
            console.print(f"[bold blue]Colunas disponíveis:[/bold blue] {', '.join(df.columns)}")

        elif user_input == 'row':
            try:
                idx = int(prompt("Digite o índice da linha: "))
                if 0 <= idx < len(df):
                    row = df.iloc[idx]
                    for key, value in row.items():
                        console.print(f"[bold]{key}:[/bold] {value}")
                else:
                    console.print("[red]Índice fora do intervalo.[/red]")
            except ValueError:
                console.print("[red]Entrada inválida.[/red]")

        elif user_input == 'search':
            col = prompt("Coluna para buscar: ").strip()
            if col not in df.columns:
                console.print("[red]Coluna não encontrada.[/red]")
                continue
            val = prompt("Valor a buscar (como string): ").strip()
            result = df[df[col].astype(str).str.contains(val, case=False, na=False)]
            print_table(result, max_rows=15)
        
        else:
            console.print("[red]Comando não reconhecido.[/red]")

if __name__ == "__main__":
    main()
