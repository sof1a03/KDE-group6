import { Component, Input } from '@angular/core';
import { UserlikeService } from '../../userlike.service';
import { UserService } from '../../user.service';

@Component({
  selector: 'app-book-card',
  standalone: true,
  imports: [],
  templateUrl: './book-card.component.html',
  styleUrl: './book-card.component.css'
})

export class BookCardComponent {
  constructor(
    private userLikeService: UserlikeService,
  private userService: UserService) {}
  @Input() image_url = '';
  @Input() title= '';
  @Input() publisher= '';
  @Input() year= 0;
  @Input() ISBN= '';
  @Input() id= '';
  @Input() genres= [''];
  @Input() author= '';

  onLike() {
    const username = this.userService.username;
    if (username !== null){
      this.userLikeService.likeBook(this.id, username);
    }
  }

}
