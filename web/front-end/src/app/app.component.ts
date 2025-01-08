import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { SidebarComponent } from "./components/sidebar/sidebar.component";
import { CommonModule } from '@angular/common';
import { BookViewComponent } from "./components/book-view/book-view.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, SidebarComponent, CommonModule, BookViewComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'front-end';
}

