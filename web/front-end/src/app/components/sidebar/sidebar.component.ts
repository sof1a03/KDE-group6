import { Component } from '@angular/core';
import { SearchMenuComponent } from "../search-menu/search-menu.component";
import { UserService } from '../../user.service';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.css',
  imports: [SearchMenuComponent, FormsModule]
})
export class SidebarComponent {
  constructor(public userService: UserService){ }
}
