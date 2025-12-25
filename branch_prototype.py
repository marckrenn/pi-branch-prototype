"""
Mock pi - TUI Prototype with Branch/Time-travel Support
Uses Textual for the terminal user interface.

Model:
- Every message is a leaf in a tree
- Each node can have multiple children
- One child per node is "selected" (the active path)
- Selecting a deep node auto-selects the whole path up to root
- Preview shows the selected trace (white) vs other options (gray)
"""

from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Header, Footer, Static, Input, Tree, ProgressBar
from textual.binding import Binding
from textual.screen import Screen
from rich.text import Text
from rich.panel import Panel
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class Message:
    """Represents a message in the conversation."""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    
    @staticmethod
    def create(role: str, content: str) -> "Message":
        return Message(
            id=str(uuid.uuid4())[:8],
            role=role,
            content=content,
            timestamp=datetime.now(),
        )


@dataclass 
class Exchange:
    """A user message and its assistant response, with children."""
    id: str
    user_msg: Message
    assistant_msg: Message
    children: list["Exchange"] = field(default_factory=list)
    selected_child_index: int = 0  # Which child is selected (0 = first)
    
    @property
    def selected_child(self) -> "Exchange | None":
        if self.children:
            return self.children[self.selected_child_index]
        return None
    
    def add_child(self, child: "Exchange") -> int:
        """Add a child and return its index."""
        self.children.append(child)
        return len(self.children) - 1
    
    def select_child(self, index: int):
        """Select a child by index."""
        if 0 <= index < len(self.children):
            self.selected_child_index = index
    
    @staticmethod
    def create(user_content: str, assistant_content: str) -> "Exchange":
        return Exchange(
            id=str(uuid.uuid4())[:8],
            user_msg=Message.create("user", user_content),
            assistant_msg=Message.create("assistant", assistant_content),
        )


class ConversationStore:
    """Stores the conversation tree."""
    
    def __init__(self):
        self.root: Exchange | None = None
        self.current_exchange_id: str | None = None  # Where we're adding from (None = end of selected path)
        
        # Initialize with mock data
        self._load_mock_data()
    
    def _load_mock_data(self):
        """Load static mock conversation data."""
        e1 = Exchange.create(
            "Can you help me write a Python function to calculate fibonacci numbers?",
            """Sure! Here's a simple recursive implementation:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

This works but is slow for large numbers due to repeated calculations."""
        )
        self.root = e1
        
        e2 = Exchange.create(
            "Can you make it more efficient?",
            """Here's an optimized version using memoization:

```python
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]
```"""
        )
        e1.children.append(e2)
        
        e3 = Exchange.create(
            "Now add type hints please",
            """Here's the iterative version with type hints:

```python
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```"""
        )
        e2.children.append(e3)
        
        # Set current position to the last message
        self.current_exchange_id = e3.id
    
    def _find_exchange(self, exchange_id: str, node: Exchange = None) -> Exchange | None:
        """Find an exchange by ID."""
        if node is None:
            node = self.root
        if node is None:
            return None
        if node.id == exchange_id:
            return node
        for child in node.children:
            found = self._find_exchange(exchange_id, child)
            if found:
                return found
        return None
    
    def _find_parent(self, exchange_id: str, node: Exchange = None, parent: Exchange = None) -> Exchange | None:
        """Find parent of an exchange."""
        if node is None:
            node = self.root
        if node is None:
            return None
        if node.id == exchange_id:
            return parent
        for child in node.children:
            found = self._find_parent(exchange_id, child, node)
            if found is not None:
                return found
        return None
    
    def get_selected_path(self) -> list[Exchange]:
        """Get the currently selected path from root to current position.
        
        If current_exchange_id is set, stops there (for branching).
        Otherwise goes to deepest selected leaf.
        """
        path = []
        current = self.root
        while current:
            path.append(current)
            # Stop if we've reached the current position
            if self.current_exchange_id and current.id == self.current_exchange_id:
                break
            if current.children:
                current = current.selected_child
            else:
                current = None
        return path
    
    def get_path_to(self, exchange_id: str) -> list[Exchange]:
        """Get path from root to specific exchange."""
        def find_path(node: Exchange, target_id: str, current_path: list[Exchange]) -> list[Exchange] | None:
            if node is None:
                return None
            new_path = current_path + [node]
            if node.id == target_id:
                return new_path
            for child in node.children:
                found = find_path(child, target_id, new_path)
                if found:
                    return found
            return None
        
        return find_path(self.root, exchange_id, []) or []
    
    def get_selected_path_ids(self) -> set[str]:
        """Get IDs of all exchanges on selected path."""
        return {ex.id for ex in self.get_selected_path()}
    
    def select_exchange(self, exchange_id: str):
        """Select an exchange, which selects the whole path to it.
        
        This also sets current_exchange_id so new messages branch from here.
        """
        path = self.get_path_to(exchange_id)
        if not path:
            return
        
        # Walk up the path and select each node in its parent
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            for j, c in enumerate(parent.children):
                if c.id == child.id:
                    parent.select_child(j)
                    break
        
        # Set current position to this exchange (for branching)
        self.current_exchange_id = exchange_id
    
    def get_current_history(self) -> list[Exchange]:
        """Get history up to current position."""
        if self.current_exchange_id:
            return self.get_path_to(self.current_exchange_id)
        return self.get_selected_path()
    
    def resume_from(self, exchange_id: str):
        """Set position to add new messages from."""
        self.current_exchange_id = exchange_id
    
    def go_to_end(self):
        """Go to end of selected path."""
        self.current_exchange_id = None
    
    def add_exchange(self, user_content: str, assistant_content: str) -> Exchange:
        """Add a new exchange as a child of current position."""
        new_exchange = Exchange.create(user_content, assistant_content)
        
        if self.root is None:
            self.root = new_exchange
        else:
            # Find where to add
            if self.current_exchange_id:
                parent = self._find_exchange(self.current_exchange_id)
            else:
                # Add to end of selected path
                path = self.get_selected_path()
                parent = path[-1] if path else None
            
            if parent:
                idx = parent.add_child(new_exchange)
                parent.select_child(idx)  # Auto-select new child
        
        # Move to new exchange and select the whole path to it
        self.current_exchange_id = new_exchange.id
        self.select_exchange(new_exchange.id)
        return new_exchange
    
    def delete_exchange(self, exchange_id: str) -> bool:
        """Delete an exchange and all its descendants."""
        if self.root and self.root.id == exchange_id:
            self.root = None
            self.current_exchange_id = None
            return True
        
        parent = self._find_parent(exchange_id)
        if parent:
            for i, child in enumerate(parent.children):
                if child.id == exchange_id:
                    parent.children.pop(i)
                    if parent.selected_child_index >= len(parent.children):
                        parent.selected_child_index = max(0, len(parent.children) - 1)
                    return True
        return False
    
    def get_child_index(self, exchange_id: str) -> int | None:
        """Get index of exchange in its parent's children."""
        parent = self._find_parent(exchange_id)
        if parent:
            for i, child in enumerate(parent.children):
                if child.id == exchange_id:
                    return i
        return None
    
    def get_context_size(self) -> int:
        """Get total character count of current selected trace."""
        total = 0
        for exchange in self.get_selected_path():
            total += len(exchange.user_msg.content)
            total += len(exchange.assistant_msg.content)
        return total
    
    # Fictional context window size (characters)
    MAX_CONTEXT_SIZE = 5000
    
    def undo(self) -> bool:
        """Go up one level in the tree (to parent of current position)."""
        path = self.get_selected_path()
        if len(path) <= 1:
            return False  # Already at root or empty
        
        # Move to parent (second to last in path)
        parent = path[-2] if len(path) >= 2 else None
        if parent:
            self.current_exchange_id = parent.id
            return True
        return False
    
    def redo(self) -> bool:
        """Go down one level in the tree (to selected child of current position)."""
        if self.current_exchange_id:
            current = self._find_exchange(self.current_exchange_id)
            if current and current.children:
                # Move to selected child
                child = current.selected_child
                if child:
                    self.current_exchange_id = child.id
                    return True
        else:
            # At end, nothing to redo
            return False
        return False
    
    def _deep_copy_exchange(self, exchange: Exchange) -> Exchange:
        """Create a deep copy of an exchange and all its children."""
        new_exchange = Exchange(
            id=str(uuid.uuid4())[:8],
            user_msg=Message(
                id=str(uuid.uuid4())[:8],
                role=exchange.user_msg.role,
                content=exchange.user_msg.content,
                timestamp=exchange.user_msg.timestamp,
            ),
            assistant_msg=Message(
                id=str(uuid.uuid4())[:8],
                role=exchange.assistant_msg.role,
                content=exchange.assistant_msg.content,
                timestamp=exchange.assistant_msg.timestamp,
            ),
            selected_child_index=exchange.selected_child_index,
        )
        for child in exchange.children:
            new_exchange.children.append(self._deep_copy_exchange(child))
        return new_exchange
    
    def start_insert_between(self, exchange_id: str):
        """Start insert-between mode at exchange_id.
        
        Saves copies of children so they can be re-attached after inserting.
        Original children stay in place.
        """
        exchange = self._find_exchange(exchange_id)
        if exchange:
            # Store deep copies of children to re-attach later
            self._insert_between_children = [self._deep_copy_exchange(c) for c in exchange.children]
            self._insert_between_parent_id = exchange_id
            # Set current position to this exchange (new messages added as new branch)
            self.current_exchange_id = exchange_id
    
    def finish_insert_between(self):
        """Finish insert-between by re-attaching saved children to the end of new branch."""
        if not hasattr(self, '_insert_between_children') or not self._insert_between_children:
            return
        
        # Find the deepest node in current selected path
        path = self.get_selected_path()
        if path:
            leaf = path[-1]
            # Attach the copied children to the leaf
            for child in self._insert_between_children:
                leaf.children.append(child)
            if leaf.children:
                leaf.selected_child_index = len(leaf.children) - len(self._insert_between_children)
        
        # Clear insert-between state
        self._insert_between_children = []
        self._insert_between_parent_id = None
        self.current_exchange_id = None
    
    def is_insert_between_mode(self) -> bool:
        """Check if we're in insert-between mode."""
        return hasattr(self, '_insert_between_children') and bool(self._insert_between_children)


class MessageWidget(Static):
    """Widget to display a single message."""
    
    def __init__(self, message: Message, **kwargs):
        super().__init__(**kwargs)
        self.message = message
    
    def compose(self) -> ComposeResult:
        role_style = "bold cyan" if self.message.role == "user" else "bold green"
        role_label = "You" if self.message.role == "user" else "Assistant"
        
        text = Text()
        text.append(f"[{self.message.id}] ", style="dim")
        text.append(f"{role_label}", style=role_style)
        text.append(f" ({self.message.timestamp.strftime('%H:%M:%S')})", style="dim")
        text.append("\n\n")
        text.append(self.message.content)
        
        yield Static(Panel(text, border_style="blue" if self.message.role == "user" else "green"))


class ChatView(ScrollableContainer):
    """Scrollable container for chat messages."""
    
    def __init__(self, store: ConversationStore, **kwargs):
        super().__init__(**kwargs)
        self.store = store
    
    def compose(self) -> ComposeResult:
        for exchange in self.store.get_current_history():
            yield MessageWidget(exchange.user_msg)
            yield MessageWidget(exchange.assistant_msg)
    
    def refresh_messages(self):
        """Refresh the message list."""
        self.remove_children()
        for exchange in self.store.get_current_history():
            self.mount(MessageWidget(exchange.user_msg))
            self.mount(MessageWidget(exchange.assistant_msg))
        self.scroll_end(animate=False)


class BranchScreen(Screen):
    """Screen showing the conversation tree with branches."""
    
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("r", "resume", "Resume/branch from here"),
        Binding("i", "insert", "Insert between"),
        Binding("s", "select", "Select this path"),
        Binding("d", "delete", "Delete (incl. below)"),
        Binding("p", "preview", "Toggle preview"),
    ]
    
    def __init__(self, store: ConversationStore, **kwargs):
        super().__init__(**kwargs)
        self.store = store
        self.highlighted_id: str | None = None
        self._node_map: dict[str, any] = {}
        self._preview_visible: bool = True
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(id="branch-header"),
            Tree("", id="branch-tree"),
            Static(id="preview-panel"),
            Container(
                ProgressBar(id="context-bar", total=self.store.MAX_CONTEXT_SIZE, show_eta=False),
                Static(id="context-info"),
                id="context-container"
            ),
            id="branch-container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        self._update_header()
        self._update_context_bar()
        self._update_preview()
        tree = self.query_one("#branch-tree", Tree)
        tree.show_root = False
        tree.root.expand()
        self._build_tree(tree)
        tree.focus()
        # Defer cursor selection until after tree is fully rendered
        self.call_after_refresh(self._select_current_node, tree)
    
    def _select_current_node(self, tree: Tree) -> None:
        """Move cursor to the current/latest node in the selected path."""
        # Get the deepest node in the selected path
        path = self.store.get_selected_path()
        if path:
            last_exchange = path[-1]
            self._select_node_by_id(tree, last_exchange.id)
    
    def _select_node_by_id(self, tree: Tree, exchange_id: str | None) -> None:
        """Move cursor to a specific node by exchange ID."""
        if exchange_id and exchange_id in self._node_map:
            node = self._node_map[exchange_id]
            tree.select_node(node)
            # Also scroll to make it visible
            tree.scroll_to_node(node)
    
    def _update_context_bar(self) -> None:
        """Update the context window progress bar."""
        size = self.store.get_context_size()
        max_size = self.store.MAX_CONTEXT_SIZE
        percentage = min(100, (size / max_size) * 100)
        
        label = self.query_one("#context-info", Static)
        bar = self.query_one("#context-bar", ProgressBar)
        
        label.update(f"Context: {size:,} / {max_size:,} chars ({percentage:.1f}%)")
        bar.update(progress=size)
    
    def _update_header(self) -> None:
        header = self.query_one("#branch-header", Static)
        if self.store.is_insert_between_mode():
            header.update(
                "[bold]Conversation Tree[/bold] [yellow](INSERT BETWEEN MODE - /done to finish)[/yellow]"
            )
        else:
            header.update("[bold]Conversation Tree[/bold]")
    
    def _build_tree(self, tree: Tree) -> None:
        """Build the tree showing conversation structure."""
        self._node_map = {}
        selected_ids = self.store.get_selected_path_ids()
        current_id = self.store.current_exchange_id
        
        def add_node(parent_node, exchange: Exchange, is_root: bool = False):
            preview = exchange.user_msg.content.replace("\n", " ")
            
            is_selected = exchange.id in selected_ids
            is_current = exchange.id == current_id
            
            # Add [>>] marker before text if this is the current branch point
            prefix = "[>>] " if is_current else ""
            
            # Style based on selection
            if is_selected:
                label = f"{prefix}{preview}"
            else:
                label = f"[#555555]{prefix}{preview}[/]"
            
            if is_root:
                # First message: add directly to tree root, expand root
                node = tree.root.add(label, data=exchange.id)
                tree.root.expand()
            else:
                node = parent_node.add(label, data=exchange.id)
            
            node.expand()
            self._node_map[exchange.id] = node
            
            for child in exchange.children:
                add_node(node, child, False)
        
        if self.store.root:
            add_node(tree.root, self.store.root, True)
        
        # Show pending insert-between children if any
        if self.store.is_insert_between_mode():
            pending_node = tree.root.add("[yellow][PENDING - type /done to reattach][/]", data=None)
            pending_node.expand()
            
            def add_pending(parent_node, exchange: Exchange):
                preview = exchange.user_msg.content.replace("\n", " ")
                label = f"[#888888]{preview}[/]"
                node = parent_node.add(label, data=None)
                node.expand()
                for child in exchange.children:
                    add_pending(node, child)
            
            for child in self.store._insert_between_children:
                add_pending(pending_node, child)
    
    def _update_tree_labels(self) -> None:
        """Update tree node labels without rebuilding the tree structure."""
        selected_ids = self.store.get_selected_path_ids()
        current_id = self.store.current_exchange_id
        
        def get_label(exchange: Exchange) -> str:
            preview = exchange.user_msg.content.replace("\n", " ")
            is_selected = exchange.id in selected_ids
            is_current = exchange.id == current_id
            prefix = "[>>] " if is_current else ""
            
            if is_selected:
                return f"{prefix}{preview}"
            else:
                return f"[#555555]{prefix}{preview}[/]"
        
        def update_node_label(exchange: Exchange):
            if exchange.id in self._node_map:
                node = self._node_map[exchange.id]
                node.set_label(get_label(exchange))
            for child in exchange.children:
                update_node_label(child)
        
        if self.store.root:
            update_node_label(self.store.root)
    
    def _refresh_tree(self, select_id: str | None = None, rebuild: bool = False):
        """Refresh the tree display.
        
        Args:
            select_id: Optional exchange ID to select after refresh.
                      If None, preserves current cursor position.
            rebuild: If True, completely rebuild the tree. If False, just update labels.
        """
        self._update_header()
        tree = self.query_one("#branch-tree", Tree)
        
        # Remember current cursor position if no explicit select_id
        if select_id is None and tree.cursor_node and tree.cursor_node.data:
            select_id = tree.cursor_node.data
        
        if rebuild:
            tree.clear()
            tree.show_root = False
            self._build_tree(tree)
            # Defer cursor selection until after tree is fully rendered
            self.call_after_refresh(self._select_node_by_id, tree, select_id)
        else:
            self._update_tree_labels()
        
        self._update_preview()
        self._update_context_bar()
    
    def _update_preview(self) -> None:
        """Update the preview panel."""
        preview_panel = self.query_one("#preview-panel", Static)
        
        if not self._preview_visible:
            preview_panel.update("")
            return
        
        selected_path = self.store.get_selected_path()
        lines = ["[bold]Preview selected context:[/bold]"]
        
        for i, exchange in enumerate(selected_path):
            preview = exchange.user_msg.content.replace("\n", " ")
            if len(preview) > 70:
                preview = preview[:70] + "..."
            lines.append(f"  {i + 1}. {preview}")
        
        preview_panel.update("\n".join(lines))
    
    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        if event.node.data:
            self.highlighted_id = event.node.data
            # Update context bar and preview to show what it would be if this was selected
            # Temporarily select to calculate, then restore
            old_current = self.store.current_exchange_id
            self.store.current_exchange_id = event.node.data
            self._update_context_bar()
            self._update_preview()
            self.store.current_exchange_id = old_current
    
    def _get_highlighted_id(self) -> str | None:
        if self.highlighted_id:
            return self.highlighted_id
        tree = self.query_one("#branch-tree", Tree)
        if tree.cursor_node and tree.cursor_node.data:
            return tree.cursor_node.data
        return None
    
    def action_close(self) -> None:
        self.app.pop_screen()
        self.app.query_one(ChatView).refresh_messages()
    
    def action_resume(self) -> None:
        """Resume/branch from selected point."""
        exchange_id = self._get_highlighted_id()
        if exchange_id:
            self.store.resume_from(exchange_id)
            self.app.pop_screen()
            self.app.query_one(ChatView).refresh_messages()
            self.app.query_one(Input).focus()
    
    def action_insert(self) -> None:
        """Insert between - saves children, lets you insert, then reattaches."""
        exchange_id = self._get_highlighted_id()
        if exchange_id:
            self.store.start_insert_between(exchange_id)
            self.app.pop_screen()
            self.app.query_one(ChatView).refresh_messages()
            # Update placeholder to show insert mode
            input_widget = self.app.query_one(Input)
            input_widget.placeholder = "INSERT MODE: Type message, then /done to finish and reattach"
            input_widget.focus()
    
    def action_select(self) -> None:
        """Select this path (makes whole path to here selected)."""
        exchange_id = self._get_highlighted_id()
        if exchange_id:
            self.store.select_exchange(exchange_id)
            self._refresh_tree(rebuild=False)  # Just update labels, preserve structure
            self.app.query_one(ChatView).refresh_messages()
    
    def action_delete(self) -> None:
        """Delete the selected exchange."""
        exchange_id = self._get_highlighted_id()
        if exchange_id:
            self.store.delete_exchange(exchange_id)
            self._refresh_tree(rebuild=True)  # Structure changed, need rebuild
            self.app.query_one(ChatView).refresh_messages()
    
    def action_preview(self) -> None:
        """Toggle preview of selected trace."""
        self._preview_visible = not self._preview_visible
        self._update_preview()


class MockPiApp(App):
    """Mock pi - TUI prototype for coding agent."""
    
    TITLE = "Mock pi"
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #chat-container {
        height: 1fr;
        border: solid green;
        margin: 1;
    }
    
    #input-container {
        height: auto;
        margin: 0 1 1 1;
    }
    
    #chat-input {
        width: 100%;
    }
    
    Footer {
        dock: bottom;
    }
    
    MessageWidget {
        margin: 0 0 1 0;
    }
    
    #branch-container {
        padding: 1 2;
    }
    
    #branch-header {
        text-align: center;
        margin-bottom: 1;
    }
    
    #context-container {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    
    #context-bar {
        width: 1fr;
    }
    
    #context-info {
        width: auto;
        text-align: right;
        margin-left: 2;
    }
    
    #branch-tree {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }
    
    #preview-panel {
        margin-top: 1;
        padding: 1;
        background: $surface-darken-1;
    }
    
    #preview-container {
        padding: 1 2;
    }
    
    #preview-header {
        margin-bottom: 1;
    }
    
    #preview-content {
        height: 1fr;
        border: solid $accent;
        padding: 1;
        background: $surface;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+b", "show_branches", "Branch View"),
        Binding("ctrl+z", "undo", "Undo"),
        Binding("ctrl+shift+z", "redo", "Redo"),
    ]
    
    def __init__(self):
        super().__init__()
        self.store = ConversationStore()
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield ChatView(self.store, id="chat-container")
        yield Container(
            Input(placeholder="Type a message... (or /branch to open branch view)", id="chat-input"),
            id="input-container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        self.query_one(ChatView).scroll_end(animate=False)
        self.query_one(Input).focus()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        
        input_widget = self.query_one(Input)
        input_widget.value = ""
        
        # Check for /branch command
        if value.lower() == "/branch":
            self.action_show_branches()
            return
        
        # Check for /done command (finish insert-between)
        if value.lower() == "/done":
            if self.store.is_insert_between_mode():
                self.store.finish_insert_between()
                self.query_one(ChatView).refresh_messages()
                self._update_input_placeholder()
            return
        
        # Add new exchange as child of current position
        chat_view = self.query_one(ChatView)
        self.store.add_exchange(
            value,
            f"This is a simulated response to: '{value}'\n\n(In a real implementation, this would call the LLM)"
        )
        
        chat_view.refresh_messages()
    
    def _update_input_placeholder(self):
        """Update input placeholder based on mode."""
        input_widget = self.query_one(Input)
        if self.store.is_insert_between_mode():
            input_widget.placeholder = "INSERT MODE: Type message, then /done to finish and reattach"
        else:
            input_widget.placeholder = "Type a message... (or /branch to open branch view)"
    
    def action_show_branches(self) -> None:
        """Show the branch tree view."""
        self.push_screen(BranchScreen(self.store))
    
    def action_undo(self) -> None:
        """Undo - go up one level in the tree."""
        if self.store.undo():
            self.query_one(ChatView).refresh_messages()
    
    def action_redo(self) -> None:
        """Redo - go down one level in the tree."""
        if self.store.redo():
            self.query_one(ChatView).refresh_messages()


if __name__ == "__main__":
    app = MockPiApp()
    app.run()
