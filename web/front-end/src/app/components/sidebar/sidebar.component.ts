import { Component } from '@angular/core';
import { SearchMenuComponent } from "../search-menu/search-menu.component";

@Component({
  selector: 'app-sidebar',
  standalone: true,
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.css',
  imports: [SearchMenuComponent]
})
export class SidebarComponent {

}
