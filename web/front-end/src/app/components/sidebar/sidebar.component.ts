import { Component } from '@angular/core';
import { SearchMenuComponent } from "../search-menu/search-menu.component";
import { UserService } from '../../user.service';
import { UserlikeService } from '../../userlike.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { OnInit } from '@angular/core';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.css',
  imports: [SearchMenuComponent, FormsModule, CommonModule]
})
export class SidebarComponent implements OnInit {
  likedBooks: { bookId: string, bookTitle: string }[] = [];
  constructor(public userService: UserService, private userLikeService: UserlikeService){ }

  ngOnInit(): void {
    const username = this.userService.username;
    if (username !== null){
      this.likedBooks = this.userLikeService.getLikedBooks(username);
      console.log(this.likedBooks)
    }
  }
  onRemoveLike(bookid: string): void {
    const username = this.userService.username;
    if (username !== null){
      this.userLikeService.removeLike(username, bookid);
      this.likedBooks = this.userLikeService.getLikedBooks(username);
    }
  }
}

